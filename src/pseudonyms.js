/**
 * Stable pseudonyms for item ids that are personal data, as an email address used as a ticket's id: the label and ten
 * hex digits of an HMAC-SHA256 of the id under a key that stays on this machine, so one id gets one pseudonym in every
 * call and every copy, and nobody without the key can tell which id it stands for by hashing guesses.
 */
import { createHmac, randomBytes } from 'node:crypto'
import { chmod, link, mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { isIPv4, isIPv6 } from 'node:net'
import { dirname } from 'node:path'
import { setTimeout as sleep } from 'node:timers/promises'

// an id shaped as an email address (its domain in any script), an IP address (v4 or v6) or a phone number is personal
// data, whether or not a span model read it; a date or a plain number (an order id) is not
const EMAIL = /^[^\s@<>()[\]",;:]+@[^\s@<>()[\]",;:]+\.(?:\p{L}{2,}|xn--[a-z0-9-]+)$/iu
const DATE = /^\d{4}-\d{2}-\d{2}(?:[T ][\d:.]+Z?)?$/
const PHONE = /^\+?[\d\s().-]+$/
// four numbers between dots: an IPv4 address when each is a byte (0 to 255), else no personal data (999.999.999.999)
const QUAD = /^\d+(?:\.\d+){3}$/

/** The label of an id that is shaped as personal data (EMAIL_ADDRESS, IP_ADDRESS, PHONE_NUMBER), or undefined. */
export function personalShape(id) {
  const value = String(id ?? '').trim()
  if (EMAIL.test(value)) return 'EMAIL_ADDRESS'
  if (isIPv4(value) || isIPv6(value.replace(/^\[(.*)\]$/, '$1'))) return 'IP_ADDRESS'
  if (QUAD.test(value) || DATE.test(value) || !PHONE.test(value)) return undefined
  const digits = value.match(/\d/g)?.length ?? 0
  return digits >= 7 && digits <= 15 && /[\s().+-]/.test(value) ? 'PHONE_NUMBER' : undefined
}

/** `label-<10 hex digits>`, the same for the same id under the same key. */
export function pseudonym(key, label, id) {
  return `${label}-${createHmac('sha256', key).update(String(id)).digest('hex').slice(0, 10)}`
}

// the key a file holds, or undefined when it holds none (missing, empty, or cut short)
async function stored(path) {
  try {
    const key = Buffer.from((await readFile(path, 'utf8')).trim(), 'hex')
    return key.length >= 32 ? key : undefined
  } catch {
    return undefined
  }
}

/**
 * The machine's pseudonym key, read from `path` (readable by its owner only), or made there on first use. A new key is
 * written whole under a name of its own and then linked to `path` only if no key is there yet, so a process starting
 * beside this one reads all of the one key both use, or makes it; a file there that holds no key (empty, or cut short
 * by a crash) is replaced. A key that cannot be kept there lasts as long as the process.
 * @returns {() => Promise<Buffer>}
 */
export function pseudonymKeys(path) {
  let key
  return () => {
    key ??= (async () => {
      for (let attempt = 0; attempt < 5; attempt++) {
        const kept = await stored(path)
        if (kept) {
          await chmod(path, 0o600).catch(() => {})
          return kept
        }
        const fresh = randomBytes(32)
        const scratch = `${path}.${randomBytes(6).toString('hex')}.new`
        try {
          await mkdir(dirname(path), { recursive: true, mode: 0o700 })
          await writeFile(scratch, `${fresh.toString('hex')}\n`, { mode: 0o600, flag: 'wx' })
          try {
            // the key's name, given only to a key written whole, and only where there is none
            await link(scratch, path)
            return fresh
          } catch (error) {
            if (error?.code !== 'EEXIST') {
              // a file system without links: the whole key under the name, once
              await rename(scratch, path)
              return (await stored(path)) ?? fresh
            }
          }
          // a file is there: another process's key, read on the next attempt, or none, which this key replaces; a key
          // a process beside this one made meanwhile is the one both read after a moment
          if (!(await stored(path))) {
            await rename(scratch, path)
            await sleep(20)
          }
        } catch {
          return fresh
        } finally {
          await rm(scratch, { force: true }).catch(() => {})
        }
      }
      return (await stored(path)) ?? randomBytes(32)
    })()
    return key
  }
}
