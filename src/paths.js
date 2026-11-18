/** The paths a System One tool call names, checked before anything is asked, so System Two can correct one at once. */
import { readdir, stat } from 'node:fs/promises'
import { dirname, extname, isAbsolute, relative, resolve } from 'node:path'
import { landing } from './permissions.js'

const SHOWN = 12

// a path as a file system that ignores letter case and Unicode normalization (Windows, macOS) names it
const folded = path => path.normalize('NFC').toLowerCase()

/**
 * Whether writing `target` would write the file at `input`: the same path, a spelling that differs only in letter case
 * or Unicode normalization (one file on Windows and macOS; refused everywhere, to be safe), a path through a link or a
 * hard link to it, or a short name (by file identity).
 */
export async function sameFile(target, input) {
  const [a, b] = [resolve(target), resolve(input)]
  if (folded(a) === folded(b)) return true
  const [landsA, landsB] = await Promise.all([landing(a), landing(b)])
  if (landsA && landsB && folded(landsA) === folded(landsB)) return true
  const [x, y] = await Promise.all([a, b].map(path => stat(path, { bigint: true }).catch(() => undefined)))
  return Boolean(x && y && x.ino !== 0n && x.dev === y.dev && x.ino === y.ino)
}

/** The first of `inputs` that writing `target` would write, or undefined. */
export async function inputAt(target, inputs) {
  for (const input of inputs) if (input && (await sameFile(target, input))) return input
  return undefined
}

/** What is at a path: 'file', 'directory', or undefined when nothing is. */
export async function kindAt(path) {
  try {
    const info = await stat(path)
    return info.isDirectory() ? 'directory' : 'file'
  } catch {
    return undefined
  }
}

/**
 * Why a path a call names is no file to read, and what the nearest folder that exists does hold: the files with the
 * same extension first, as "metrics/charts/ holds active-users.png, api-error-rate.png, … (16)".
 */
export async function noFile(cwd, path, what) {
  const full = resolve(cwd, path)
  if ((await kindAt(full)) === 'directory') return `${what} ${path} is a folder, not a file${await holding(cwd, full, extname(full))}`
  let folder = dirname(full)
  while ((await kindAt(folder)) !== 'directory' && dirname(folder) !== folder) folder = dirname(folder)
  return `${what} ${path} does not exist${await holding(cwd, folder, extname(full))}`
}

async function holding(cwd, folder, ext) {
  // only the working directory's own folders are listed
  const inside = relative(cwd, folder)
  if (inside.startsWith('..') || isAbsolute(inside)) return ''
  let entries
  try {
    entries = await readdir(folder, { withFileTypes: true })
  } catch {
    return ''
  }
  const names = entries.filter(entry => !entry.name.startsWith('.')).map(entry => `${entry.name}${entry.isDirectory() ? '/' : ''}`).sort()
  const alike = ext ? names.filter(name => extname(name).toLowerCase() === ext.toLowerCase()) : []
  const listed = alike.length ? alike : names
  const where = inside ? `${inside}/` : 'the working directory'
  if (listed.length === 0) return `; ${where} is empty`
  return `; ${where} holds ${listed.slice(0, SHOWN).join(', ')}${listed.length > SHOWN ? `, … (${listed.length})` : ''}`
}

/** Throws noFile's reason unless the path is a file. */
export async function mustBeFile(cwd, path, what) {
  if ((await kindAt(resolve(cwd, path))) !== 'file') throw new Error(await noFile(cwd, path, what))
}
