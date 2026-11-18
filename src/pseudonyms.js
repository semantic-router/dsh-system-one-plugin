/**
 * Stable pseudonyms for item ids that are personal data, as an email address used as a ticket's id: the label and ten
 * hex digits of an HMAC-SHA256 of the id under a key that stays on this machine, so one id gets one pseudonym in every
 * call and every copy, and nobody without the key can tell which id it stands for by hashing guesses.
 */
import { createHmac, randomBytes } from 'node:crypto'
import { chmod, mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

// an id shaped as an email address, an IP address or a phone number is personal data, whether or not a span model
// read it; a date or a plain number (an order id) is not
const EMAIL = /^[^\s@<>()[\]",;:]+@[^\s@<>()[\]",;:]+\.[A-Za-z]{2,}$/
const IPV4 = /^(?:\d{1,3}\.){3}\d{1,3}$/
const DATE = /^\d{4}-\d{2}-\d{2}(?:[T ][\d:.]+Z?)?$/
const PHONE = /^\+?[\d\s().-]+$/

/** The label of an id that is shaped as personal data (EMAIL_ADDRESS, IP_ADDRESS, PHONE_NUMBER), or undefined. */
export function personalShape(id) {
  const value = String(id ?? '').trim()
  if (EMAIL.test(value)) return 'EMAIL_ADDRESS'
  if (IPV4.test(value)) return 'IP_ADDRESS'
  if (DATE.test(value) || !PHONE.test(value)) return undefined
  const digits = value.match(/\d/g)?.length ?? 0
  return digits >= 7 && digits <= 15 && /[\s().+-]/.test(value) ? 'PHONE_NUMBER' : undefined
}

/** `label-<10 hex digits>`, the same for the same id under the same key. */
export function pseudonym(key, label, id) {
  return `${label}-${createHmac('sha256', key).update(String(id)).digest('hex').slice(0, 10)}`
}

async function stored(path) {
  try {
    const key = Buffer.from((await readFile(path, 'utf8')).trim(), 'hex')
    return key.length >= 32 ? key : undefined
  } catch {
    return undefined
  }
}

/**
 * The machine's pseudonym key, read from `path` (readable by its owner only), or made there on first use. A key that
 * cannot be kept there lasts as long as the process.
 * @returns {() => Promise<Buffer>}
 */
export function pseudonymKeys(path) {
  let key
  return () => {
    key ??= (async () => {
      const kept = await stored(path)
      if (kept) {
        await chmod(path, 0o600).catch(() => {})
        return kept
      }
      const fresh = randomBytes(32)
      try {
        await mkdir(dirname(path), { recursive: true, mode: 0o700 })
        await writeFile(path, `${fresh.toString('hex')}\n`, { mode: 0o600, flag: 'wx' })
      } catch (error) {
        // another process made it first: that one is the machine's
        if (error?.code === 'EEXIST') return (await stored(path)) ?? fresh
      }
      return fresh
    })()
    return key
  }
}
