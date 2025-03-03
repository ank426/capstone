-- Use this to get it working inside neovim
return {
  "mfussenegger/nvim-lint",
  config = function()
    -- path/to/file:line: severity: message
    local pattern = '([^:]+):(%d+): (%a+): (.*)'
    local groups = {'file', 'lnum', 'severity', 'message'}
    local severities = {
      error = vim.diagnostic.severity.ERROR,
      warning = vim.diagnostic.severity.WARN,
      note = vim.diagnostic.severity.HINT,
    }
    require("lint").linters.capstone = {
      name = "capstone",
      cmd = vim.fn.expand("~/code/pes/capstone/frontend/script.sh"),
      append_fname = true,
      parser = require("lint.parser").from_pattern(pattern, groups, severities),
    }
  end,
}
