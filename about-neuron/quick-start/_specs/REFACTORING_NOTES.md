# Quick-Start Refactoring Notes

## Summary

The quick-start documentation has been restructured with a modern, task-based information architecture. The new structure eliminates the need for .txt includes in the primary quickstart paths.

## New Structure (No .txt includes)

### Primary Quickstarts (Self-contained)
- `index.rst` - Main landing page with decision tree
- `training-quickstart.rst` - Complete training workflow (no includes)
- `inference-quickstart.rst` - Complete inference workflow (no includes)

These files follow the procedural-quickstart template and contain all content inline. No external includes required.

### Supporting Pages
- `docs-quicklinks.rst` - Quick navigation links
- `github-samples.rst` - GitHub repository links

## Legacy Structure (Uses .txt includes)

### Legacy Quick-Start Pages (Inf1 only)
- `torch-neuron.rst` - Uses tab-inference-torch-neuronx.txt and tab-inference-torch-neuron.txt
- `tensorflow-neuron.rst` - Uses tab-inference-tensorflow-neuronx.txt and tab-inference-tensorflow-neuron.rst
- `mxnet-neuron.rst` - Uses tab-inference-mxnet-neuron.txt

These legacy pages:
- Target Inf1 instances (NeuronCore v1)
- Use .txt includes that reference `/src/helperscripts/installationScripts/python_instructions.txt`
- Are de-emphasized in the new navigation (under "Legacy" section)
- Are preserved for backward compatibility and existing links

### .txt Include Files (Legacy only)
All .txt files in this directory are used exclusively by the legacy quick-start pages:
- `tab-inference-torch-neuronx*.txt` (various OS versions)
- `tab-inference-torch-neuron*.txt` (various OS versions)
- `tab-inference-tensorflow-neuronx*.txt` (various OS versions)
- `tab-inference-tensorflow-neuron*.txt` (various OS versions)
- `tab-inference-mxnet-neuron*.txt` (various OS versions)
- `select-framework-note.txt`

## Design Decision

**Why not refactor legacy files?**
1. They target deprecated Inf1 hardware
2. They're not prominently featured in new navigation
3. Refactoring would require updating installation script references
4. Risk of breaking existing external links
5. New users are directed to the new self-contained quickstarts

**Why are new quickstarts self-contained?**
1. Easier to maintain (all content in one place)
2. Better for AI/LLM context retrieval
3. Follows modern docs-as-code best practices
4. Clearer for human readers (no jumping between files)
5. Follows the procedural-quickstart template structure

## Migration Path

For users currently using legacy quick-starts:
- Inf1 users: Continue using legacy pages (torch-neuron.rst, etc.)
- New projects: Use new quickstarts (training-quickstart.rst, inference-quickstart.rst)
- Inf2/Trn1/Trn2/Trn3 users: Use new quickstarts

## Future Cleanup

When Inf1 support is fully deprecated:
1. Archive legacy quick-start pages to `/archive/quick-start/`
2. Remove .txt include files
3. Update any remaining cross-references
4. Update neuron_tag.py to remove special handling
