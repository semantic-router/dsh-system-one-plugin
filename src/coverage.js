/**
 * Which of DeepSeek Harness's tools each reflex reads, by what the tools do rather than by a few names: the shield
 * screens every result that brings in text from files, shells, the web or an MCP server; the guard reviews every call
 * that runs a command, writes a file, sends data out, plans a prompt for later or acts through an MCP server. In each
 * list, a name that ends in `*` stands for every tool whose name starts with the rest (MCP tools are
 * `mcp__<server>__<tool>`).
 */

// file text (read; grep's matching lines; glob's file names; str_replace_editor's view), shell text (bash; pwsh,
// Windows' shell; job_output, a background job's), web text (web_fetch, web_search) and an MCP server's
// (read_mcp_resource, and every MCP tool)
export const SHIELD_TOOLS = ['read', 'grep', 'glob', 'str_replace_editor', 'bash', 'pwsh', 'job_output', 'web_fetch', 'web_search', 'read_mcp_resource', 'mcp__*']

// commands (bash, pwsh), file writes (write, edit, apply_patch, str_replace_editor), data sent out (web_fetch's
// address, web_search's queries), prompts planned for later (schedule_create, schedule_update), and MCP tools
export const GUARD_TOOLS = ['bash', 'pwsh', 'write', 'edit', 'apply_patch', 'str_replace_editor', 'web_fetch', 'web_search', 'schedule_create', 'schedule_update', 'mcp__*']

// the text of files, shells and the web, where personal data comes in; glob's file names and MCP tools only when
// named, as a masked name in a path would leave System Two unable to open the file
export const PRIVACY_TOOLS = ['web_fetch', 'bash', 'read', 'pwsh', 'job_output', 'grep', 'web_search', 'str_replace_editor']

// the tools whose results DeepSeek Harness writes lines of its own into (a stream's marker, a sandbox note): a shell's
// output and a background job's; in any other tool's result such a line is the text's own (shield.js `harnessLines`)
export const HARNESS_NOTED = new Set(['bash', 'pwsh', 'job_output'])

/** A test of a tool's name against a list of names, where `name*` stands for every name that starts with `name`. */
export function toolMatcher(names) {
  const exact = new Set()
  const prefixes = []
  for (const name of names ?? []) {
    const value = String(name)
    if (value.endsWith('*')) prefixes.push(value.slice(0, -1))
    else exact.add(value)
  }
  return name => typeof name === 'string' && (exact.has(name) || prefixes.some(prefix => name.startsWith(prefix)))
}

/** A call of a guarded tool that only reads, so the guard lets it run unasked: str_replace_editor's view. */
export const onlyReads = exec => exec?.name === 'str_replace_editor' && exec.arguments?.command === 'view'

const short = (value, max) => {
  const text = typeof value === 'string' ? value : JSON.stringify(value ?? '')
  return text.length <= max ? text : `${text.slice(0, max)}…`
}

/** Where a tool result's text came from, as the shield's question tells System One. */
export function sourceOf(exec) {
  const args = exec?.arguments ?? {}
  switch (exec?.name) {
    case 'web_fetch': return `a web page (${short(args.url, 300)})`
    case 'web_search': return `web search results for ${short(args.queries ?? args.query, 200)}`
    case 'read': return `the file ${short(args.file_path, 300)}`
    case 'str_replace_editor': return `the file ${short(args.path, 300)}`
    case 'grep': return `a search of the files for ${short(args.pattern, 200)}`
    case 'glob': return `the names of the files that match ${short(args.pattern, 200)}`
    case 'job_output': return 'the output of a background job'
    case 'read_mcp_resource': return `an MCP server's resource ${short(args.uri, 300)}`
    case 'bash':
    case 'pwsh': return `the output of: ${short(args.command, 200)}`
    default: return String(exec?.name ?? '').startsWith('mcp__') ? `the result of the MCP tool ${exec.name}` : `the result of the tool ${exec?.name}`
  }
}
