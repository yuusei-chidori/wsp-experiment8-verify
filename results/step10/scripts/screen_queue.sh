#!/bin/zsh
cd "${0:A:h}/.."
S=./venv/bin/python
$S scripts/screen.py base0 '[{"kind":"wait","layer_dir":19,"layer_edit":19,"alpha":0}]'
$S scripts/screen.py waitL19a1 '[{"kind":"wait","layer_dir":19,"layer_edit":19,"alpha":1}]'
$S scripts/screen.py waitL19a3 '[{"kind":"wait","layer_dir":19,"layer_edit":19,"alpha":3}]'
$S scripts/screen.py wrongL15a1 '[{"kind":"wrong","layer_dir":15,"layer_edit":15,"alpha":1}]'
$S scripts/screen.py wrongL15a3 '[{"kind":"wrong","layer_dir":15,"layer_edit":15,"alpha":3}]'
$S scripts/screen.py wrongL15L19a6 '[{"kind":"wrong","layer_dir":15,"layer_edit":15,"alpha":6},{"kind":"wrong","layer_dir":19,"layer_edit":19,"alpha":6}]'
echo SCREEN_DONE
