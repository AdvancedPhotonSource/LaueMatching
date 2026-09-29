"""Column content, Laue reference implementation: which crystal orientations are present in one beam column
(interaction volume), each with its intensity share and orientation spread, and how completely that list explains the
frame. Method, interfaces, validated ranges and traps: MIDAS manuals/column-content/. The monochromatic-rotation (DAC)
implementation of the same method is midas_defect.column_content.

Modules: geom (projection, measured kernel, cloud statistics), fit (joint mixture ColumnFit), pipeline (rounds with an
injected indexer; measured discovery gate), synthetic (known-content columns), evaluate (frozen gates), arcs
(beaded-arc and common-axis diagnostics for what the fit leaves unexplained).
"""
