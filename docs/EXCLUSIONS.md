# File exclusions

One local exclusion policy is used by file listing, literal search, repository RAG,
reference indexing, dependency analysis, and direct file reads. CodeHaven never writes files.
An excluded path cannot be retrieved by giving the model its exact name. No Git
executable, Git repository, network service, or model call is needed for filtering.

## Git-style files

At the selected workspace root, and in allowed subdirectories, the agent reads
`.gitignore` and `.agentignore`. Rules use Git-style syntax: comments, `*`, `?`,
`**`, rooted patterns, directory-only rules, escaped special characters, and `!`
negations. Patterns are relative to the directory containing the rule file.
Later matching rules in a layer win; a nested rule file can override a parent
file's rule for a file in that subtree.

For example, `.agentignore` can contain:

```gitignore
# Keep these visible project files out of the agent's context.
private/
*.pem
reports/*
!reports/public.md
```

As with Git, a child cannot be re-included while its parent directory is excluded.
Use `reports/*` rather than `reports/` if you want to re-include `reports/public.md`.

## Extra command-line exclusions

```sh
offline-agent --workspace /path/to/repo \
  --exclude 'private/**' --exclude '*.pem' \
  'Explain the authentication flow'
```

Repeat `--exclude` for multiple Git-style patterns. Quote globs so your shell does
not expand them. CLI patterns are deny-only: a leading `!` is rejected. Use ordered
rules in `.agentignore` for exceptions within that layer.

Layer precedence is deliberately restrictive:

1. Hidden paths, built-in excluded directories, symlinks, and workspace boundaries
   remain blocked and cannot be re-included.
2. A path ignored by `.gitignore` is blocked.
3. `.agentignore` can exclude additional paths; its negations do not override Git
   ignores or built-in restrictions.
4. CLI exclusions add further blocks and cannot be undone by rule files.

Git-style patterns apply even to tracked files; this is an agent content filter,
not Git's tracked-file status behavior. Only rule files within each configured
root are read. Parent rules above `--workspace`, global Git excludes, Git config,
and `.git/info/exclude` are not consulted. Select the repository root if you want
its root `.gitignore` to apply to all your source files.

## Reference libraries and RAG

The same CLI patterns apply relative to each indexing root, including built-in
references and external libraries. Each library also has its own `.gitignore` and
`.agentignore` rules. A reference library inside the workspace additionally
inherits the workspace's rules; using `--knowledge-base` does not bypass them.
Excluded reference roots are counted in retrieval metadata and reported at startup.

If a project has `knowledge/` in `.gitignore`, that folder is not indexed. To keep
private references out of the repository while making them available locally,
place them outside the workspace:

```sh
offline-agent --workspace /path/to/repo \
  --knowledge-base /path/to/my-private-reference-library \
  'Use my coding standards to review the design'
```

This repository already ignores `knowledge/`. Other projects can use the default
`workspace/knowledge/` location if their exclusion rules permit it.

## Limits and refresh

Rule files must be regular UTF-8 files up to 64 KB. Symlinked rule files are not
followed. Unreadable, oversized, or invalid rule files stop the affected listing/
indexing operation rather than silently exposing files. Rule files are read as
configuration, not added to model context as hidden documents.

Policies are cached for a workspace instance. Restart a task after externally
editing ignore files. RAG rebuilds its policies/index for each task,
preserving CLI exclusions. Already retrieved conversation text is not erased by a
later rule change. Filtering alone is not an OS security boundary. Test execution now uses a filtered,
read-only snapshot inside Docker with networking disabled; see [isolated tests](ISOLATED_TESTS.md).
Existing per-run approval requirements remain unchanged.
