/**
 * Where a System One tool may write files: where DeepSeek Harness lets its own write tool write in the calling
 * session. Its sandboxed filesystem (`ctx.fs` with `ctx.sandboxPolicy`) writes nothing in `read-only`, writes under the
 * session's workspace, its extra writable folders and the temporary folders in `workspace-write`, and anywhere in
 * `danger-full-access`; a harness whose filesystem confines nothing confines these tools neither.
 */
import { lstat, realpath, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, dirname, isAbsolute, join, resolve, sep } from 'node:path'

/**
 * The calling session's write policy, from the harness's services.
 * @param {{sandboxPolicy?: {resolve: Function}, fs?: {sandboxMode?: string}}} services
 * @param {object} [session] - the calling agent's session
 * @returns {{mode: 'read-only' | 'workspace-write', workspace?: string, roots: string[], unknown?: string} | undefined}
 *   undefined when writes are not confined; a policy that cannot be read writes nothing
 */
export function writePolicy({ sandboxPolicy, fs } = {}, session) {
  if (typeof sandboxPolicy?.resolve !== 'function' || (fs && fs.sandboxMode === undefined)) return undefined
  let policy
  try {
    policy = sandboxPolicy.resolve(session ? { session } : {})
  } catch (error) {
    return { mode: 'read-only', roots: [], unknown: String(error?.message ?? error) }
  }
  if (policy?.mode === 'danger-full-access') return undefined
  if (policy?.mode !== 'workspace-write') return { mode: 'read-only', roots: [] }
  const roots = [policy.workspaceRoot, ...(Array.isArray(policy.additionalWritableRoots) ? policy.additionalWritableRoots : []), '/tmp', tmpdir()]
  return { mode: 'workspace-write', workspace: policy.workspaceRoot, roots: roots.filter(root => typeof root === 'string' && isAbsolute(root)) }
}

const MISSING = new Set(['ENOENT', 'ENOTDIR'])

/**
 * Where a write to `path` lands: its deepest existing ancestor with every link resolved, and the rest of the path. A
 * link whose target does not exist, which a write would follow, lands nowhere it can say.
 */
export async function landing(path) {
  const rest = []
  for (let at = resolve(path); ; ) {
    try {
      return join(await realpath(at), ...rest)
    } catch (error) {
      if (!MISSING.has(error?.code)) return undefined
      try {
        if ((await lstat(at)).isSymbolicLink()) return undefined
      } catch {}
      const parent = dirname(at)
      if (parent === at) return undefined
      rest.unshift(basename(at))
      at = parent
    }
  }
}

// a root and what is under it; a spelling that differs (a Windows short name, letter case) is matched by file identity
async function under(path, root) {
  const fold = value => (process.platform === 'win32' ? value.toLowerCase() : value)
  const [target, base] = [fold(path), fold(root)]
  if (target === base || target.startsWith(base.endsWith(sep) ? base : base + sep)) return true
  const rootInfo = await stat(root, { bigint: true }).catch(() => undefined)
  if (!rootInfo) return false
  for (let at = path; ; at = dirname(at)) {
    const info = await stat(at, { bigint: true }).catch(() => undefined)
    if (info && info.dev === rootInfo.dev && info.ino === rootInfo.ino) return true
    if (dirname(at) === at) return false
  }
}

/** Whether `policy` lets a tool write `path`. */
export async function mayWrite(policy, path) {
  if (!policy) return true
  if (policy.roots.length === 0) return false
  const target = await landing(path)
  if (!target) return false
  for (const root of policy.roots) {
    if (await under(target, (await landing(root)) ?? root)) return true
  }
  return false
}

/** Why `policy` keeps a tool from writing `shown`, for System Two to choose another path or ask the user. */
export function deniedText(policy, what, shown) {
  if (policy.mode !== 'workspace-write') return `${what} ${shown} cannot be written: this session's file policy is read-only`
  return `${what} ${shown} is outside this session's workspace (${policy.workspace}), where DeepSeek Harness's file policy (workspace-write) lets tools write: name a path inside it, or ask the user to allow writing there`
}
