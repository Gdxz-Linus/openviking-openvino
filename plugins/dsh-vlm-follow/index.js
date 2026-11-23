/**
 * dsh-vlm-follow
 *
 * 让 OpenViking 的记忆抽取 VLM 跟随 DSH 当前实际使用的模型通道。
 *
 * 与 Hermes 的 hermes-vlm-follow 插件对称：
 *   Hermes: pre_api_request 钩子 → 写 ov.conf 的 vlm 段
 *   DSH:    llm/stream waterfall → 写 ov.conf 的 vlm 段
 *
 * OpenViking 服务端每 30 秒重载配置文件并 diff（REFRESH_INTERVAL_SECS=30），
 * 检测到 vlm 变化就重建 VLM 实例，所以写文件即可生效，无需调 API、无需重启。
 *
 * 失败一律吞掉：记忆抽取配置同步失败绝不能影响正在进行的对话。
 */

import { readFileSync, writeFileSync, renameSync, existsSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { createRequire } from 'node:module'

export const name = 'dsh-vlm-follow'
export const inject = ['llm']

const require = createRequire(import.meta.url)

// js-yaml 通常由 host 提供；解析不到时退回内置的最小解析器。
let yaml = null
try {
  yaml = require('js-yaml')
} catch {
  try {
    yaml = require('yaml')
  } catch {
    yaml = null
  }
}

// ── 路径解析 ────────────────────────────────────────────────────────────────

/** OpenViking 服务端配置。环境变量优先，否则用 D 盘生产部署的默认位置。 */
function ovConfPath() {
  const fromEnv = (process.env.OPENVIKING_CONFIG_FILE || '').trim()
  return fromEnv || 'D:/OpenViking/ov.conf'
}

/** DSH profile 目录，插件与 patch 都在这里。 */
function profileDir() {
  const fromEnv = (process.env.DSH_PROFILE_DIR || '').trim()
  return fromEnv || join(homedir(), '.dsh', 'profiles', 'web')
}

/** DSH 凭据文件。 */
function credentialsPath() {
  const fromEnv = (process.env.DSH_HOME || '').trim()
  const home = fromEnv || join(homedir(), '.dsh')
  return join(home, '.credentials.yaml')
}

// ── 配置读取 ────────────────────────────────────────────────────────────────

/**
 * 从 profile 的 patch 里找某个 provider 的 baseURL 与 apiKeyEnv。
 *
 * 只读用户的 cordis.patch.yml：DSH 的 provider 定义就写在那里，
 * 是唯一权威来源（bundle 层只会定义内置路由，不会覆盖用户的 baseURL）。
 */
function resolveProvider(providerName) {
  const patchPath = join(profileDir(), 'cordis.patch.yml')
  let text
  try {
    text = readFileSync(patchPath, 'utf8')
  } catch {
    return null
  }

  let doc = null
  if (yaml) {
    try {
      doc = yaml.load ? yaml.load(text) : yaml.parse(text)
    } catch {
      doc = null
    }
  }

  if (Array.isArray(doc)) {
    for (const entry of doc) {
      const config = entry && entry.config
      const providers = config && config.providers
      if (!providers || typeof providers !== 'object') continue
      const hit = providers[providerName]
      if (!hit || typeof hit !== 'object') continue
      return {
        baseURL: typeof hit.baseURL === 'string' ? hit.baseURL : (typeof hit.base_url === 'string' ? hit.base_url : ''),
        apiKeyEnv: typeof hit.apiKeyEnv === 'string' ? hit.apiKeyEnv : (typeof hit.api_key_env === 'string' ? hit.api_key_env : ''),
      }
    }
    return null
  }

  // ── fallback：无 YAML 库时按缩进做最小解析 ──
  return scanProvidersByIndent(text, providerName)
}

function scanProvidersByIndent(text, providerName) {
  const lines = text.split(/\r?\n/)
  let providersIndent = -1
  let nameIndent = -1
  let inTarget = false
  const out = { baseURL: '', apiKeyEnv: '' }

  for (const raw of lines) {
    const trimmed = raw.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    const indent = raw.length - raw.trimStart().length

    if (/^providers:\s*$/.test(trimmed)) {
      providersIndent = indent
      nameIndent = -1
      inTarget = false
      continue
    }
    if (providersIndent < 0) continue
    if (indent <= providersIndent) {
      // 离开 providers 块
      if (inTarget) break
      providersIndent = -1
      continue
    }

    // providers 下第一个更深缩进的键即 provider 名
    if (nameIndent < 0) nameIndent = indent

    if (indent === nameIndent) {
      const m = trimmed.match(/^([\w.\-]+):\s*$/)
      inTarget = Boolean(m && m[1] === providerName)
      continue
    }

    if (inTarget && indent > nameIndent) {
      const m = trimmed.match(/^(baseURL|base_url|apiKeyEnv|api_key_env):\s*(.+)$/)
      if (m) {
        const key = m[1]
        const value = m[2].trim().replace(/^["']|["']$/g, '')
        if (key === 'baseURL' || key === 'base_url') out.baseURL = value
        else out.apiKeyEnv = value
      }
    }
  }
  return out.baseURL || out.apiKeyEnv ? out : null
}

/** 取 API key：进程环境优先，其次 .credentials.yaml 的 refs 段。 */
function resolveApiKey(envName) {
  if (!envName) return ''
  const fromEnv = (process.env[envName] || '').trim()
  if (fromEnv) return fromEnv

  let text
  try {
    text = readFileSync(credentialsPath(), 'utf8')
  } catch {
    return ''
  }
  // refs 段下的 `名: 值` 行；值可能带单双引号。
  const re = new RegExp(`^\\s*${envName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*:\\s*(.+?)\\s*$`, 'm')
  const m = text.match(re)
  return m ? m[1].replace(/^["']|["']$/g, '').trim() : ''
}

// ── 写入 ov.conf ────────────────────────────────────────────────────────────

/**
 * 把 vlm 段写成给定值。返回值表示是否真的改了。
 *
 * 幂等：三要素都相同时不落盘，热路径零 I/O 写。
 * 原子：先写 .tmp 再 rename，避免 OpenViking 读到半个 JSON。
 */
function writeVlm(baseUrl, model, apiKey) {
  const path = ovConfPath()
  if (!existsSync(path)) return false

  let conf
  try {
    conf = JSON.parse(readFileSync(path, 'utf8'))
  } catch {
    return false
  }
  if (!conf || typeof conf !== 'object') return false

  const current = conf.vlm && typeof conf.vlm === 'object' ? conf.vlm : {}
  if (
    current.api_base === baseUrl &&
    current.model === model &&
    (apiKey ? current.api_key === apiKey : true)
  ) {
    return false
  }

  conf.vlm = {
    provider: 'openai',
    model,
    api_base: baseUrl,
    api_key: apiKey || current.api_key || '',
  }

  const tmp = `${path}.tmp`
  try {
    writeFileSync(tmp, JSON.stringify(conf, null, 2), 'utf8')
    renameSync(tmp, path)
    return true
  } catch {
    try { writeFileSync(tmp, '', 'utf8') } catch { /* ignore */ }
    return false
  }
}

// ── 插件主体 ────────────────────────────────────────────────────────────────

export function apply(ctx) {
  // 短路缓存的另一半：上一次同步的组合 + 写入后的文件 mtime。
  // 只比对 key 不够——ov.conf 是 Hermes 与 DSH 共写的共享文件，
  // 对方改了之后本进程必须能察觉并纠正，否则配置会静默漂移。
  let lastKey = ''
  let lastMtime = 0

  function currentMtime(path) {
    try {
      return statSync(path).mtimeMs
    } catch {
      return 0
    }
  }

  function sync(providerName, modelName) {
    if (!providerName || !modelName) return
    const path = ovConfPath()
    const key = `${providerName}\u0000${modelName}`
    const mtime = currentMtime(path)

    // 通道未变且文件未被他人改动 → 无无需任何读取
    if (key === lastKey && mtime !== 0 && mtime === lastMtime) return

    const provider = resolveProvider(providerName)
    if (!provider || !provider.baseURL) return

    const apiKey = resolveApiKey(provider.apiKeyEnv)
    const changed = writeVlm(provider.baseURL, modelName, apiKey)
    lastKey = key
    lastMtime = currentMtime(path)
    if (changed) {
      try {
        ctx.logger?.info?.(
          `dsh-vlm-follow: 记忆抽取已跟随到 ${provider.baseURL} / ${modelName}`,
        )
      } catch { /* logging must never break the path */ }
    }
  }

  // llm/stream 是 cordis waterfall：必须返回 next()，且不能阻塞请求。
  ctx.on('llm/stream', (options, next) => {
    try {
      sync(options?.provider, options?.model)
    } catch {
      // 同步失败绝不阻断对话；下一个请求会重试。
    }
    return next()
  })
}
