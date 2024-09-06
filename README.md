<!DOCTYPE html>

<html lang="en">
<head><meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>CNN_Keras_Project_Cell_Imaging</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<style type="text/css">
    pre { line-height: 125%; }
td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
.highlight .hll { background-color: var(--jp-cell-editor-active-background) }
.highlight { background: var(--jp-cell-editor-background); color: var(--jp-mirror-editor-variable-color) }
.highlight .c { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment */
.highlight .err { color: var(--jp-mirror-editor-error-color) } /* Error */
.highlight .k { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword */
.highlight .o { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator */
.highlight .p { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation */
.highlight .ch { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Multiline */
.highlight .cp { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Preproc */
.highlight .cpf { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Single */
.highlight .cs { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Special */
.highlight .kc { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Pseudo */
.highlight .kr { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Type */
.highlight .m { color: var(--jp-mirror-editor-number-color) } /* Literal.Number */
.highlight .s { color: var(--jp-mirror-editor-string-color) } /* Literal.String */
.highlight .ow { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator.Word */
.highlight .pm { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation.Marker */
.highlight .w { color: var(--jp-mirror-editor-variable-color) } /* Text.Whitespace */
.highlight .mb { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Bin */
.highlight .mf { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Float */
.highlight .mh { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Hex */
.highlight .mi { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer */
.highlight .mo { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Oct */
.highlight .sa { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Affix */
.highlight .sb { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Backtick */
.highlight .sc { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Char */
.highlight .dl { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Delimiter */
.highlight .sd { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Doc */
.highlight .s2 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Double */
.highlight .se { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Escape */
.highlight .sh { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Heredoc */
.highlight .si { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Interpol */
.highlight .sx { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Other */
.highlight .sr { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Regex */
.highlight .s1 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Single */
.highlight .ss { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Symbol */
.highlight .il { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer.Long */
  </style>
<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
 * Mozilla scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */
[data-jp-theme-scrollbars='true'] {
  scrollbar-color: rgb(var(--jp-scrollbar-thumb-color))
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar. These selectors
 * will match lower in the tree, and so will override the above */
[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
  scrollbar-width: thin;
}

/* tiny scrollbar */

.jp-scrollbar-tiny::-webkit-scrollbar,
.jp-scrollbar-tiny::-webkit-scrollbar-corner {
  background-color: transparent;
  height: 4px;
  width: 4px;
}

.jp-scrollbar-tiny::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:horizontal {
  border-left: 0 solid transparent;
  border-right: 0 solid transparent;
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:vertical {
  border-top: 0 solid transparent;
  border-bottom: 0 solid transparent;
}

/*
 * Lumino
 */

.lm-ScrollBar[data-orientation='horizontal'] {
  min-height: 16px;
  max-height: 16px;
  min-width: 45px;
  border-top: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] {
  min-width: 16px;
  max-width: 16px;
  min-height: 45px;
  border-left: 1px solid #a0a0a0;
}

.lm-ScrollBar-button {
  background-color: #f0f0f0;
  background-position: center center;
  min-height: 15px;
  max-height: 15px;
  min-width: 15px;
  max-width: 15px;
}

.lm-ScrollBar-button:hover {
  background-color: #dadada;
}

.lm-ScrollBar-button.lm-mod-active {
  background-color: #cdcdcd;
}

.lm-ScrollBar-track {
  background: #f0f0f0;
}

.lm-ScrollBar-thumb {
  background: #cdcdcd;
}

.lm-ScrollBar-thumb:hover {
  background: #bababa;
}

.lm-ScrollBar-thumb.lm-mod-active {
  background: #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal'] .lm-ScrollBar-thumb {
  height: 100%;
  min-width: 15px;
  border-left: 1px solid #a0a0a0;
  border-right: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] .lm-ScrollBar-thumb {
  width: 100%;
  min-height: 15px;
  border-top: 1px solid #a0a0a0;
  border-bottom: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-left);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-right);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-up);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-down);
  background-size: 17px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-Widget {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
}

.lm-Widget.lm-mod-hidden {
  display: none !important;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.lm-AccordionPanel[data-orientation='horizontal'] > .lm-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  display: flex;
  flex-direction: column;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-CommandPalette-search {
  flex: 0 0 auto;
}

.lm-CommandPalette-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  min-height: 0;
  overflow: auto;
  list-style-type: none;
}

.lm-CommandPalette-header {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-CommandPalette-item {
  display: flex;
  flex-direction: row;
}

.lm-CommandPalette-itemIcon {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemContent {
  flex: 1 1 auto;
  overflow: hidden;
}

.lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemLabel {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-close-icon {
  border: 1px solid transparent;
  background-color: transparent;
  position: absolute;
  z-index: 1;
  right: 3%;
  top: 0;
  bottom: 0;
  margin: auto;
  padding: 7px 0;
  display: none;
  vertical-align: middle;
  outline: 0;
  cursor: pointer;
}
.lm-close-icon:after {
  content: 'X';
  display: block;
  width: 15px;
  height: 15px;
  text-align: center;
  color: #000;
  font-weight: normal;
  font-size: 12px;
  cursor: pointer;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-DockPanel {
  z-index: 0;
}

.lm-DockPanel-widget {
  z-index: 0;
}

.lm-DockPanel-tabBar {
  z-index: 1;
}

.lm-DockPanel-handle {
  z-index: 2;
}

.lm-DockPanel-handle.lm-mod-hidden {
  display: none !important;
}

.lm-DockPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

.lm-DockPanel-handle[data-orientation='horizontal'] {
  cursor: ew-resize;
}

.lm-DockPanel-handle[data-orientation='vertical'] {
  cursor: ns-resize;
}

.lm-DockPanel-handle[data-orientation='horizontal']:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

.lm-DockPanel-handle[data-orientation='vertical']:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

.lm-DockPanel-overlay {
  z-index: 3;
  box-sizing: border-box;
  pointer-events: none;
}

.lm-DockPanel-overlay.lm-mod-hidden {
  display: none !important;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-Menu {
  z-index: 10000;
  position: absolute;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: auto;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-Menu-content {
  margin: 0;
  padding: 0;
  display: table;
  list-style-type: none;
}

.lm-Menu-item {
  display: table-row;
}

.lm-Menu-item.lm-mod-hidden,
.lm-Menu-item.lm-mod-collapsed {
  display: none !important;
}

.lm-Menu-itemIcon,
.lm-Menu-itemSubmenuIcon {
  display: table-cell;
  text-align: center;
}

.lm-Menu-itemLabel {
  display: table-cell;
  text-align: left;
}

.lm-Menu-itemShortcut {
  display: table-cell;
  text-align: right;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-MenuBar {
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-MenuBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: row;
  list-style-type: none;
}

.lm-MenuBar-item {
  box-sizing: border-box;
}

.lm-MenuBar-itemIcon,
.lm-MenuBar-itemLabel {
  display: inline-block;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-ScrollBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-ScrollBar[data-orientation='horizontal'] {
  flex-direction: row;
}

.lm-ScrollBar[data-orientation='vertical'] {
  flex-direction: column;
}

.lm-ScrollBar-button {
  box-sizing: border-box;
  flex: 0 0 auto;
}

.lm-ScrollBar-track {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  flex: 1 1 auto;
}

.lm-ScrollBar-thumb {
  box-sizing: border-box;
  position: absolute;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-SplitPanel-child {
  z-index: 0;
}

.lm-SplitPanel-handle {
  z-index: 1;
}

.lm-SplitPanel-handle.lm-mod-hidden {
  display: none !important;
}

.lm-SplitPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
  cursor: ew-resize;
}

.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
  cursor: ns-resize;
}

.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
  align-items: flex-end;
}

.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
  align-items: flex-end;
}

.lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}

.lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
  flex-direction: row;
}

.lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
  flex-direction: column;
}

.lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
  touch-action: none; /* Disable native Drag/Drop */
}

.lm-TabBar-tabIcon,
.lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}

.lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}

.lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
}

.lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar-addButton.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}

.lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}

.lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}

.lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

.lm-TabBar-tabLabel .lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
  background: inherit;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-TabPanel-tabBar {
  z-index: 1;
}

.lm-TabPanel-stackedPanel {
  z-index: 0;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapse {
  display: flex;
  flex-direction: column;
  align-items: stretch;
}

.jp-Collapse-header {
  padding: 1px 12px;
  background-color: var(--jp-layout-color1);
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  align-items: center;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  text-transform: uppercase;
  user-select: none;
}

.jp-Collapser-icon {
  height: 16px;
}

.jp-Collapse-header-collapsed .jp-Collapser-icon {
  transform: rotate(-90deg);
  margin: auto 0;
}

.jp-Collapser-title {
  line-height: 25px;
}

.jp-Collapse-contents {
  padding: 0 12px;
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensureUiComponents() in @jupyterlab/buildutils */

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

/* Icons urls */

:root {
  --jp-icon-add-above: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzEzN18xOTQ5MikiPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGQ9Ik00Ljc1IDQuOTMwNjZINi42MjVWNi44MDU2NkM2LjYyNSA3LjAxMTkxIDYuNzkzNzUgNy4xODA2NiA3IDcuMTgwNjZDNy4yMDYyNSA3LjE4MDY2IDcuMzc1IDcuMDExOTEgNy4zNzUgNi44MDU2NlY0LjkzMDY2SDkuMjVDOS40NTYyNSA0LjkzMDY2IDkuNjI1IDQuNzYxOTEgOS42MjUgNC41NTU2NkM5LjYyNSA0LjM0OTQxIDkuNDU2MjUgNC4xODA2NiA5LjI1IDQuMTgwNjZINy4zNzVWMi4zMDU2NkM3LjM3NSAyLjA5OTQxIDcuMjA2MjUgMS45MzA2NiA3IDEuOTMwNjZDNi43OTM3NSAxLjkzMDY2IDYuNjI1IDIuMDk5NDEgNi42MjUgMi4zMDU2NlY0LjE4MDY2SDQuNzVDNC41NDM3NSA0LjE4MDY2IDQuMzc1IDQuMzQ5NDEgNC4zNzUgNC41NTU2NkM0LjM3NSA0Ljc2MTkxIDQuNTQzNzUgNC45MzA2NiA0Ljc1IDQuOTMwNjZaIiBmaWxsPSIjNjE2MTYxIiBzdHJva2U9IiM2MTYxNjEiIHN0cm9rZS13aWR0aD0iMC43Ii8+CjwvZz4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTExLjUgOS41VjExLjVMMi41IDExLjVWOS41TDExLjUgOS41Wk0xMiA4QzEyLjU1MjMgOCAxMyA4LjQ0NzcyIDEzIDlWMTJDMTMgMTIuNTUyMyAxMi41NTIzIDEzIDEyIDEzTDIgMTNDMS40NDc3MiAxMyAxIDEyLjU1MjMgMSAxMlY5QzEgOC40NDc3MiAxLjQ0NzcxIDggMiA4TDEyIDhaIiBmaWxsPSIjNjE2MTYxIi8+CjxkZWZzPgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzEzN18xOTQ5MiI+CjxyZWN0IGNsYXNzPSJqcC1pY29uMyIgd2lkdGg9IjYiIGhlaWdodD0iNiIgZmlsbD0id2hpdGUiIHRyYW5zZm9ybT0ibWF0cml4KC0xIDAgMCAxIDEwIDEuNTU1NjYpIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg==);
  --jp-icon-add-below: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzEzN18xOTQ5OCkiPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGQ9Ik05LjI1IDEwLjA2OTNMNy4zNzUgMTAuMDY5M0w3LjM3NSA4LjE5NDM0QzcuMzc1IDcuOTg4MDkgNy4yMDYyNSA3LjgxOTM0IDcgNy44MTkzNEM2Ljc5Mzc1IDcuODE5MzQgNi42MjUgNy45ODgwOSA2LjYyNSA4LjE5NDM0TDYuNjI1IDEwLjA2OTNMNC43NSAxMC4wNjkzQzQuNTQzNzUgMTAuMDY5MyA0LjM3NSAxMC4yMzgxIDQuMzc1IDEwLjQ0NDNDNC4zNzUgMTAuNjUwNiA0LjU0Mzc1IDEwLjgxOTMgNC43NSAxMC44MTkzTDYuNjI1IDEwLjgxOTNMNi42MjUgMTIuNjk0M0M2LjYyNSAxMi45MDA2IDYuNzkzNzUgMTMuMDY5MyA3IDEzLjA2OTNDNy4yMDYyNSAxMy4wNjkzIDcuMzc1IDEyLjkwMDYgNy4zNzUgMTIuNjk0M0w3LjM3NSAxMC44MTkzTDkuMjUgMTAuODE5M0M5LjQ1NjI1IDEwLjgxOTMgOS42MjUgMTAuNjUwNiA5LjYyNSAxMC40NDQzQzkuNjI1IDEwLjIzODEgOS40NTYyNSAxMC4wNjkzIDkuMjUgMTAuMDY5M1oiIGZpbGw9IiM2MTYxNjEiIHN0cm9rZT0iIzYxNjE2MSIgc3Ryb2tlLXdpZHRoPSIwLjciLz4KPC9nPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMi41IDUuNUwyLjUgMy41TDExLjUgMy41TDExLjUgNS41TDIuNSA1LjVaTTIgN0MxLjQ0NzcyIDcgMSA2LjU1MjI4IDEgNkwxIDNDMSAyLjQ0NzcyIDEuNDQ3NzIgMiAyIDJMMTIgMkMxMi41NTIzIDIgMTMgMi40NDc3MiAxMyAzTDEzIDZDMTMgNi41NTIyOSAxMi41NTIzIDcgMTIgN0wyIDdaIiBmaWxsPSIjNjE2MTYxIi8+CjxkZWZzPgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzEzN18xOTQ5OCI+CjxyZWN0IGNsYXNzPSJqcC1pY29uMyIgd2lkdGg9IjYiIGhlaWdodD0iNiIgZmlsbD0id2hpdGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMS43NDg0NmUtMDcgMS43NDg0NmUtMDcgLTEgNCAxMy40NDQzKSIvPgo8L2NsaXBQYXRoPgo8L2RlZnM+Cjwvc3ZnPgo=);
  --jp-icon-add: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDEzaC02djZoLTJ2LTZINXYtMmg2VjVoMnY2aDZ2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bell: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE2IDE2IiB2ZXJzaW9uPSIxLjEiPgogICA8cGF0aCBjbGFzcz0ianAtaWNvbjIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMzMzMzMzIgogICAgICBkPSJtOCAwLjI5Yy0xLjQgMC0yLjcgMC43My0zLjYgMS44LTEuMiAxLjUtMS40IDMuNC0xLjUgNS4yLTAuMTggMi4yLTAuNDQgNC0yLjMgNS4zbDAuMjggMS4zaDVjMC4wMjYgMC42NiAwLjMyIDEuMSAwLjcxIDEuNSAwLjg0IDAuNjEgMiAwLjYxIDIuOCAwIDAuNTItMC40IDAuNi0xIDAuNzEtMS41aDVsMC4yOC0xLjNjLTEuOS0wLjk3LTIuMi0zLjMtMi4zLTUuMy0wLjEzLTEuOC0wLjI2LTMuNy0xLjUtNS4yLTAuODUtMS0yLjItMS44LTMuNi0xLjh6bTAgMS40YzAuODggMCAxLjkgMC41NSAyLjUgMS4zIDAuODggMS4xIDEuMSAyLjcgMS4yIDQuNCAwLjEzIDEuNyAwLjIzIDMuNiAxLjMgNS4yaC0xMGMxLjEtMS42IDEuMi0zLjQgMS4zLTUuMiAwLjEzLTEuNyAwLjMtMy4zIDEuMi00LjQgMC41OS0wLjcyIDEuNi0xLjMgMi41LTEuM3ptLTAuNzQgMTJoMS41Yy0wLjAwMTUgMC4yOCAwLjAxNSAwLjc5LTAuNzQgMC43OS0wLjczIDAuMDAxNi0wLjcyLTAuNTMtMC43NC0wLjc5eiIgLz4KPC9zdmc+Cg==);
  --jp-icon-bug-dot: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiPgogICAgICAgIDxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMTcuMTkgOEgyMFYxMEgxNy45MUMxNy45NiAxMC4zMyAxOCAxMC42NiAxOCAxMVYxMkgyMFYxNEgxOC41SDE4VjE0LjAyNzVDMTUuNzUgMTQuMjc2MiAxNCAxNi4xODM3IDE0IDE4LjVDMTQgMTkuMjA4IDE0LjE2MzUgMTkuODc3OSAxNC40NTQ5IDIwLjQ3MzlDMTMuNzA2MyAyMC44MTE3IDEyLjg3NTcgMjEgMTIgMjFDOS43OCAyMSA3Ljg1IDE5Ljc5IDYuODEgMThINFYxNkg2LjA5QzYuMDQgMTUuNjcgNiAxNS4zNCA2IDE1VjE0SDRWMTJINlYxMUM2IDEwLjY2IDYuMDQgMTAuMzMgNi4wOSAxMEg0VjhINi44MUM3LjI2IDcuMjIgNy44OCA2LjU1IDguNjIgNi4wNEw3IDQuNDFMOC40MSAzTDEwLjU5IDUuMTdDMTEuMDQgNS4wNiAxMS41MSA1IDEyIDVDMTIuNDkgNSAxMi45NiA1LjA2IDEzLjQyIDUuMTdMMTUuNTkgM0wxNyA0LjQxTDE1LjM3IDYuMDRDMTYuMTIgNi41NSAxNi43NCA3LjIyIDE3LjE5IDhaTTEwIDE2SDE0VjE0SDEwVjE2Wk0xMCAxMkgxNFYxMEgxMFYxMloiIGZpbGw9IiM2MTYxNjEiLz4KICAgICAgICA8cGF0aCBkPSJNMjIgMTguNUMyMiAyMC40MzMgMjAuNDMzIDIyIDE4LjUgMjJDMTYuNTY3IDIyIDE1IDIwLjQzMyAxNSAxOC41QzE1IDE2LjU2NyAxNi41NjcgMTUgMTguNSAxNUMyMC40MzMgMTUgMjIgMTYuNTY3IDIyIDE4LjVaIiBmaWxsPSIjNjE2MTYxIi8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bug: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yMCA4aC0yLjgxYy0uNDUtLjc4LTEuMDctMS40NS0xLjgyLTEuOTZMMTcgNC40MSAxNS41OSAzbC0yLjE3IDIuMTdDMTIuOTYgNS4wNiAxMi40OSA1IDEyIDVjLS40OSAwLS45Ni4wNi0xLjQxLjE3TDguNDEgMyA3IDQuNDFsMS42MiAxLjYzQzcuODggNi41NSA3LjI2IDcuMjIgNi44MSA4SDR2MmgyLjA5Yy0uMDUuMzMtLjA5LjY2LS4wOSAxdjFINHYyaDJ2MWMwIC4zNC4wNC42Ny4wOSAxSDR2MmgyLjgxYzEuMDQgMS43OSAyLjk3IDMgNS4xOSAzczQuMTUtMS4yMSA1LjE5LTNIMjB2LTJoLTIuMDljLjA1LS4zMy4wOS0uNjYuMDktMXYtMWgydi0yaC0ydi0xYzAtLjM0LS4wNC0uNjctLjA5LTFIMjBWOHptLTYgOGgtNHYtMmg0djJ6bTAtNGgtNHYtMmg0djJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-build: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE0LjkgMTcuNDVDMTYuMjUgMTcuNDUgMTcuMzUgMTYuMzUgMTcuMzUgMTVDMTcuMzUgMTMuNjUgMTYuMjUgMTIuNTUgMTQuOSAxMi41NUMxMy41NCAxMi41NSAxMi40NSAxMy42NSAxMi40NSAxNUMxMi40NSAxNi4zNSAxMy41NCAxNy40NSAxNC45IDE3LjQ1Wk0yMC4xIDE1LjY4TDIxLjU4IDE2Ljg0QzIxLjcxIDE2Ljk1IDIxLjc1IDE3LjEzIDIxLjY2IDE3LjI5TDIwLjI2IDE5LjcxQzIwLjE3IDE5Ljg2IDIwIDE5LjkyIDE5LjgzIDE5Ljg2TDE4LjA5IDE5LjE2QzE3LjczIDE5LjQ0IDE3LjMzIDE5LjY3IDE2LjkxIDE5Ljg1TDE2LjY0IDIxLjdDMTYuNjIgMjEuODcgMTYuNDcgMjIgMTYuMyAyMkgxMy41QzEzLjMyIDIyIDEzLjE4IDIxLjg3IDEzLjE1IDIxLjdMMTIuODkgMTkuODVDMTIuNDYgMTkuNjcgMTIuMDcgMTkuNDQgMTEuNzEgMTkuMTZMOS45NjAwMiAxOS44NkM5LjgxMDAyIDE5LjkyIDkuNjIwMDIgMTkuODYgOS41NDAwMiAxOS43MUw4LjE0MDAyIDE3LjI5QzguMDUwMDIgMTcuMTMgOC4wOTAwMiAxNi45NSA4LjIyMDAyIDE2Ljg0TDkuNzAwMDIgMTUuNjhMOS42NTAwMSAxNUw5LjcwMDAyIDE0LjMxTDguMjIwMDIgMTMuMTZDOC4wOTAwMiAxMy4wNSA4LjA1MDAyIDEyLjg2IDguMTQwMDIgMTIuNzFMOS41NDAwMiAxMC4yOUM5LjYyMDAyIDEwLjEzIDkuODEwMDIgMTAuMDcgOS45NjAwMiAxMC4xM0wxMS43MSAxMC44NEMxMi4wNyAxMC41NiAxMi40NiAxMC4zMiAxMi44OSAxMC4xNUwxMy4xNSA4LjI4OTk4QzEzLjE4IDguMTI5OTggMTMuMzIgNy45OTk5OCAxMy41IDcuOTk5OThIMTYuM0MxNi40NyA3Ljk5OTk4IDE2LjYyIDguMTI5OTggMTYuNjQgOC4yODk5OEwxNi45MSAxMC4xNUMxNy4zMyAxMC4zMiAxNy43MyAxMC41NiAxOC4wOSAxMC44NEwxOS44MyAxMC4xM0MyMCAxMC4wNyAyMC4xNyAxMC4xMyAyMC4yNiAxMC4yOUwyMS42NiAxMi43MUMyMS43NSAxMi44NiAyMS43MSAxMy4wNSAyMS41OCAxMy4xNkwyMC4xIDE0LjMxTDIwLjE1IDE1TDIwLjEgMTUuNjhaIi8+CiAgICA8cGF0aCBkPSJNNy4zMjk2NiA3LjQ0NDU0QzguMDgzMSA3LjAwOTU0IDguMzM5MzIgNi4wNTMzMiA3LjkwNDMyIDUuMjk5ODhDNy40NjkzMiA0LjU0NjQzIDYuNTA4MSA0LjI4MTU2IDUuNzU0NjYgNC43MTY1NkM1LjM5MTc2IDQuOTI2MDggNS4xMjY5NSA1LjI3MTE4IDUuMDE4NDkgNS42NzU5NEM0LjkxMDA0IDYuMDgwNzEgNC45NjY4MiA2LjUxMTk4IDUuMTc2MzQgNi44NzQ4OEM1LjYxMTM0IDcuNjI4MzIgNi41NzYyMiA3Ljg3OTU0IDcuMzI5NjYgNy40NDQ1NFpNOS42NTcxOCA0Ljc5NTkzTDEwLjg2NzIgNC45NTE3OUMxMC45NjI4IDQuOTc3NDEgMTEuMDQwMiA1LjA3MTMzIDExLjAzODIgNS4xODc5M0wxMS4wMzg4IDYuOTg4OTNDMTEuMDQ1NSA3LjEwMDU0IDEwLjk2MTYgNy4xOTUxOCAxMC44NTUgNy4yMTA1NEw5LjY2MDAxIDcuMzgwODNMOS4yMzkxNSA4LjEzMTg4TDkuNjY5NjEgOS4yNTc0NUM5LjcwNzI5IDkuMzYyNzEgOS42NjkzNCA5LjQ3Njk5IDkuNTc0MDggOS41MzE5OUw4LjAxNTIzIDEwLjQzMkM3LjkxMTMxIDEwLjQ5MiA3Ljc5MzM3IDEwLjQ2NzcgNy43MjEwNSAxMC4zODI0TDYuOTg3NDggOS40MzE4OEw2LjEwOTMxIDkuNDMwODNMNS4zNDcwNCAxMC4zOTA1QzUuMjg5MDkgMTAuNDcwMiA1LjE3MzgzIDEwLjQ5MDUgNS4wNzE4NyAxMC40MzM5TDMuNTEyNDUgOS41MzI5M0MzLjQxMDQ5IDkuNDc2MzMgMy4zNzY0NyA5LjM1NzQxIDMuNDEwNzUgOS4yNTY3OUwzLjg2MzQ3IDguMTQwOTNMMy42MTc0OSA3Ljc3NDg4TDMuNDIzNDcgNy4zNzg4M0wyLjIzMDc1IDcuMjEyOTdDMi4xMjY0NyA3LjE5MjM1IDIuMDQwNDkgNy4xMDM0MiAyLjA0MjQ1IDYuOTg2ODJMMi4wNDE4NyA1LjE4NTgyQzIuMDQzODMgNS4wNjkyMiAyLjExOTA5IDQuOTc5NTggMi4yMTcwNCA0Ljk2OTIyTDMuNDIwNjUgNC43OTM5M0wzLjg2NzQ5IDQuMDI3ODhMMy40MTEwNSAyLjkxNzMxQzMuMzczMzcgMi44MTIwNCAzLjQxMTMxIDIuNjk3NzYgMy41MTUyMyAyLjYzNzc2TDUuMDc0MDggMS43Mzc3NkM1LjE2OTM0IDEuNjgyNzYgNS4yODcyOSAxLjcwNzA0IDUuMzU5NjEgMS43OTIzMUw2LjExOTE1IDIuNzI3ODhMNi45ODAwMSAyLjczODkzTDcuNzI0OTYgMS43ODkyMkM3Ljc5MTU2IDEuNzA0NTggNy45MTU0OCAxLjY3OTIyIDguMDA4NzkgMS43NDA4Mkw5LjU2ODIxIDIuNjQxODJDOS42NzAxNyAyLjY5ODQyIDkuNzEyODUgMi44MTIzNCA5LjY4NzIzIDIuOTA3OTdMOS4yMTcxOCA0LjAzMzgzTDkuNDYzMTYgNC4zOTk4OEw5LjY1NzE4IDQuNzk1OTNaIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iOS45LDEzLjYgMy42LDcuNCA0LjQsNi42IDkuOSwxMi4yIDE1LjQsNi43IDE2LjEsNy40ICIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNS45TDksOS43bDMuOC0zLjhsMS4yLDEuMmwtNC45LDVsLTQuOS01TDUuMiw1Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNy41TDksMTEuMmwzLjgtMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-left: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik0xMC44LDEyLjhMNy4xLDlsMy44LTMuOGwwLDcuNkgxMC44eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-right: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik03LjIsNS4yTDEwLjksOWwtMy44LDMuOFY1LjJINy4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-up-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iMTUuNCwxMy4zIDkuOSw3LjcgNC40LDEzLjIgMy42LDEyLjUgOS45LDYuMyAxNi4xLDEyLjYgIi8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-up: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik01LjIsMTAuNUw5LDYuOGwzLjgsMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-case-sensitive: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWFjY2VudDIiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTcuNiw4aDAuOWwzLjUsOGgtMS4xTDEwLDE0SDZsLTAuOSwySDRMNy42LDh6IE04LDkuMUw2LjQsMTNoMy4yTDgsOS4xeiIvPgogICAgPHBhdGggZD0iTTE2LjYsOS44Yy0wLjIsMC4xLTAuNCwwLjEtMC43LDAuMWMtMC4yLDAtMC40LTAuMS0wLjYtMC4yYy0wLjEtMC4xLTAuMi0wLjQtMC4yLTAuNyBjLTAuMywwLjMtMC42LDAuNS0wLjksMC43Yy0wLjMsMC4xLTAuNywwLjItMS4xLDAuMmMtMC4zLDAtMC41LDAtMC43LTAuMWMtMC4yLTAuMS0wLjQtMC4yLTAuNi0wLjNjLTAuMi0wLjEtMC4zLTAuMy0wLjQtMC41IGMtMC4xLTAuMi0wLjEtMC40LTAuMS0wLjdjMC0wLjMsMC4xLTAuNiwwLjItMC44YzAuMS0wLjIsMC4zLTAuNCwwLjQtMC41QzEyLDcsMTIuMiw2LjksMTIuNSw2LjhjMC4yLTAuMSwwLjUtMC4xLDAuNy0wLjIgYzAuMy0wLjEsMC41LTAuMSwwLjctMC4xYzAuMiwwLDAuNC0wLjEsMC42LTAuMWMwLjIsMCwwLjMtMC4xLDAuNC0wLjJjMC4xLTAuMSwwLjItMC4yLDAuMi0wLjRjMC0xLTEuMS0xLTEuMy0xIGMtMC40LDAtMS40LDAtMS40LDEuMmgtMC45YzAtMC40LDAuMS0wLjcsMC4yLTFjMC4xLTAuMiwwLjMtMC40LDAuNS0wLjZjMC4yLTAuMiwwLjUtMC4zLDAuOC0wLjNDMTMuMyw0LDEzLjYsNCwxMy45LDQgYzAuMywwLDAuNSwwLDAuOCwwLjFjMC4zLDAsMC41LDAuMSwwLjcsMC4yYzAuMiwwLjEsMC40LDAuMywwLjUsMC41QzE2LDUsMTYsNS4yLDE2LDUuNnYyLjljMCwwLjIsMCwwLjQsMCwwLjUgYzAsMC4xLDAuMSwwLjIsMC4zLDAuMmMwLjEsMCwwLjIsMCwwLjMsMFY5Ljh6IE0xNS4yLDYuOWMtMS4yLDAuNi0zLjEsMC4yLTMuMSwxLjRjMCwxLjQsMy4xLDEsMy4xLTAuNVY2Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik05IDE2LjE3TDQuODMgMTJsLTEuNDIgMS40MUw5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-circle-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDJDNi40NyAyIDIgNi40NyAyIDEyczQuNDcgMTAgMTAgMTAgMTAtNC40NyAxMC0xMFMxNy41MyAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iOSIgY3k9IjkiIHI9IjgiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-clear: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8bWFzayBpZD0iZG9udXRIb2xlIj4KICAgIDxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0id2hpdGUiIC8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI4IiBmaWxsPSJibGFjayIvPgogIDwvbWFzaz4KCiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxyZWN0IGhlaWdodD0iMTgiIHdpZHRoPSIyIiB4PSIxMSIgeT0iMyIgdHJhbnNmb3JtPSJyb3RhdGUoMzE1LCAxMiwgMTIpIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgbWFzaz0idXJsKCNkb251dEhvbGUpIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-close: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1ub25lIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIGpwLWljb24zLWhvdmVyIiBmaWxsPSJub25lIj4KICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjExIi8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIGpwLWljb24tYWNjZW50Mi1ob3ZlciIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMTkgNi40MUwxNy41OSA1IDEyIDEwLjU5IDYuNDEgNSA1IDYuNDEgMTAuNTkgMTIgNSAxNy41OSA2LjQxIDE5IDEyIDEzLjQxIDE3LjU5IDE5IDE5IDE3LjU5IDEzLjQxIDEyeiIvPgogIDwvZz4KCiAgPGcgY2xhc3M9ImpwLWljb24tbm9uZSBqcC1pY29uLWJ1c3kiIGZpbGw9Im5vbmUiPgogICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBzaGFwZS1yZW5kZXJpbmc9Imdlb21ldHJpY1ByZWNpc2lvbiI+CiAgICA8cGF0aCBkPSJNNi41OSwzLjQxTDIsOEw2LjU5LDEyLjZMOCwxMS4xOEw0LjgyLDhMOCw0LjgyTDYuNTksMy40MU0xMi40MSwzLjQxTDExLDQuODJMMTQuMTgsOEwxMSwxMS4xOEwxMi40MSwxMi42TDE3LDhMMTIuNDEsMy40MU0yMS41OSwxMS41OUwxMy41LDE5LjY4TDkuODMsMTZMOC40MiwxNy40MUwxMy41LDIyLjVMMjMsMTNMMjEuNTksMTEuNTlaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTExLjQgMTguNkw2LjggMTRMMTEuNCA5LjRMMTAgOEw0IDE0TDEwIDIwTDExLjQgMTguNlpNMTYuNiAxOC42TDIxLjIgMTRMMTYuNiA5LjRMMTggOEwyNCAxNEwxOCAyMEwxNi42IDE4LjZWMTguNloiLz4KCTwvZz4KPC9zdmc+Cg==);
  --jp-icon-collapse-all: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTggMmMxIDAgMTEgMCAxMiAwczIgMSAyIDJjMCAxIDAgMTEgMCAxMnMwIDItMiAyQzIwIDE0IDIwIDQgMjAgNFMxMCA0IDYgNGMwLTIgMS0yIDItMnoiIC8+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTE4IDhjMC0xLTEtMi0yLTJTNSA2IDQgNnMtMiAxLTIgMmMwIDEgMCAxMSAwIDEyczEgMiAyIDJjMSAwIDExIDAgMTIgMHMyLTEgMi0yYzAtMSAwLTExIDAtMTJ6bS0yIDB2MTJINFY4eiIgLz4KICAgICAgICA8cGF0aCBkPSJNNiAxM3YyaDh2LTJ6IiAvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-console: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwMCAyMDAiPgogIDxnIGNsYXNzPSJqcC1jb25zb2xlLWljb24tYmFja2dyb3VuZC1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMjg4RDEiPgogICAgPHBhdGggZD0iTTIwIDE5LjhoMTYwdjE1OS45SDIweiIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtY29uc29sZS1pY29uLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIj4KICAgIDxwYXRoIGQ9Ik0xMDUgMTI3LjNoNDB2MTIuOGgtNDB6TTUxLjEgNzdMNzQgOTkuOWwtMjMuMyAyMy4zIDEwLjUgMTAuNSAyMy4zLTIzLjNMOTUgOTkuOSA4NC41IDg5LjQgNjEuNiA2Ni41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copy: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTExLjksMUgzLjJDMi40LDEsMS43LDEuNywxLjcsMi41djEwLjJoMS41VjIuNWg4LjdWMXogTTE0LjEsMy45aC04Yy0wLjgsMC0xLjUsMC43LTEuNSwxLjV2MTAuMmMwLDAuOCwwLjcsMS41LDEuNSwxLjVoOCBjMC44LDAsMS41LTAuNywxLjUtMS41VjUuNEMxNS41LDQuNiwxNC45LDMuOSwxNC4xLDMuOXogTTE0LjEsMTUuNWgtOFY1LjRoOFYxNS41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copyright: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDI0IDI0IiBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCI+CiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0xMS44OCw5LjE0YzEuMjgsMC4wNiwxLjYxLDEuMTUsMS42MywxLjY2aDEuNzljLTAuMDgtMS45OC0xLjQ5LTMuMTktMy40NS0zLjE5QzkuNjQsNy42MSw4LDksOCwxMi4xNCBjMCwxLjk0LDAuOTMsNC4yNCwzLjg0LDQuMjRjMi4yMiwwLDMuNDEtMS42NSwzLjQ0LTIuOTVoLTEuNzljLTAuMDMsMC41OS0wLjQ1LDEuMzgtMS42MywxLjQ0QzEwLjU1LDE0LjgzLDEwLDEzLjgxLDEwLDEyLjE0IEMxMCw5LjI1LDExLjI4LDkuMTYsMTEuODgsOS4xNHogTTEyLDJDNi40OCwyLDIsNi40OCwyLDEyczQuNDgsMTAsMTAsMTBzMTAtNC40OCwxMC0xMFMxNy41MiwyLDEyLDJ6IE0xMiwyMGMtNC40MSwwLTgtMy41OS04LTggczMuNTktOCw4LThzOCwzLjU5LDgsOFMxNi40MSwyMCwxMiwyMHoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-cut: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkuNjQgNy42NGMuMjMtLjUuMzYtMS4wNS4zNi0xLjY0IDAtMi4yMS0xLjc5LTQtNC00UzIgMy43OSAyIDZzMS43OSA0IDQgNGMuNTkgMCAxLjE0LS4xMyAxLjY0LS4zNkwxMCAxMmwtMi4zNiAyLjM2QzcuMTQgMTQuMTMgNi41OSAxNCA2IDE0Yy0yLjIxIDAtNCAxLjc5LTQgNHMxLjc5IDQgNCA0IDQtMS43OSA0LTRjMC0uNTktLjEzLTEuMTQtLjM2LTEuNjRMMTIgMTRsNyA3aDN2LTFMOS42NCA3LjY0ek02IDhjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTAgMTJjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTYtNy41Yy0uMjggMC0uNS0uMjItLjUtLjVzLjIyLS41LjUtLjUuNS4yMi41LjUtLjIyLjUtLjUuNXpNMTkgM2wtNiA2IDIgMiA3LTdWM3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-delete: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2cHgiIGhlaWdodD0iMTZweCI+CiAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIiAvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjI2MjYyIiBkPSJNNiAxOWMwIDEuMS45IDIgMiAyaDhjMS4xIDAgMi0uOSAyLTJWN0g2djEyek0xOSA0aC0zLjVsLTEtMWgtNWwtMSAxSDV2MmgxNFY0eiIgLz4KPC9zdmc+Cg==);
  --jp-icon-download: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDloLTRWM0g5djZINWw3IDcgNy03ek01IDE4djJoMTR2LTJINXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-duplicate: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTIuNzk5OTggMC44NzVIOC44OTU4MkM5LjIwMDYxIDAuODc1IDkuNDQ5OTggMS4xMzkxNCA5LjQ0OTk4IDEuNDYxOThDOS40NDk5OCAxLjc4NDgyIDkuMjAwNjEgMi4wNDg5NiA4Ljg5NTgyIDIuMDQ4OTZIMy4zNTQxNUMzLjA0OTM2IDIuMDQ4OTYgMi43OTk5OCAyLjMxMzEgMi43OTk5OCAyLjYzNTk0VjkuNjc5NjlDMi43OTk5OCAxMC4wMDI1IDIuNTUwNjEgMTAuMjY2NyAyLjI0NTgyIDEwLjI2NjdDMS45NDEwMyAxMC4yNjY3IDEuNjkxNjUgMTAuMDAyNSAxLjY5MTY1IDkuNjc5NjlWMi4wNDg5NkMxLjY5MTY1IDEuNDAzMjggMi4xOTA0IDAuODc1IDIuNzk5OTggMC44NzVaTTUuMzY2NjUgMTEuOVY0LjU1SDExLjA4MzNWMTEuOUg1LjM2NjY1Wk00LjE0MTY1IDQuMTQxNjdDNC4xNDE2NSAzLjY5MDYzIDQuNTA3MjggMy4zMjUgNC45NTgzMiAzLjMyNUgxMS40OTE3QzExLjk0MjcgMy4zMjUgMTIuMzA4MyAzLjY5MDYzIDEyLjMwODMgNC4xNDE2N1YxMi4zMDgzQzEyLjMwODMgMTIuNzU5NCAxMS45NDI3IDEzLjEyNSAxMS40OTE3IDEzLjEyNUg0Ljk1ODMyQzQuNTA3MjggMTMuMTI1IDQuMTQxNjUgMTIuNzU5NCA0LjE0MTY1IDEyLjMwODNWNC4xNDE2N1oiIGZpbGw9IiM2MTYxNjEiLz4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNOS40MzU3NCA4LjI2NTA3SDguMzY0MzFWOS4zMzY1QzguMzY0MzEgOS40NTQzNSA4LjI2Nzg4IDkuNTUwNzggOC4xNTAwMiA5LjU1MDc4QzguMDMyMTcgOS41NTA3OCA3LjkzNTc0IDkuNDU0MzUgNy45MzU3NCA5LjMzNjVWOC4yNjUwN0g2Ljg2NDMxQzYuNzQ2NDUgOC4yNjUwNyA2LjY1MDAyIDguMTY4NjQgNi42NTAwMiA4LjA1MDc4QzYuNjUwMDIgNy45MzI5MiA2Ljc0NjQ1IDcuODM2NSA2Ljg2NDMxIDcuODM2NUg3LjkzNTc0VjYuNzY1MDdDNy45MzU3NCA2LjY0NzIxIDguMDMyMTcgNi41NTA3OCA4LjE1MDAyIDYuNTUwNzhDOC4yNjc4OCA2LjU1MDc4IDguMzY0MzEgNi42NDcyMSA4LjM2NDMxIDYuNzY1MDdWNy44MzY1SDkuNDM1NzRDOS41NTM2IDcuODM2NSA5LjY1MDAyIDcuOTMyOTIgOS42NTAwMiA4LjA1MDc4QzkuNjUwMDIgOC4xNjg2NCA5LjU1MzYgOC4yNjUwNyA5LjQzNTc0IDguMjY1MDdaIiBmaWxsPSIjNjE2MTYxIiBzdHJva2U9IiM2MTYxNjEiIHN0cm9rZS13aWR0aD0iMC41Ii8+Cjwvc3ZnPgo=);
  --jp-icon-edit: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMgMTcuMjVWMjFoMy43NUwxNy44MSA5Ljk0bC0zLjc1LTMuNzVMMyAxNy4yNXpNMjAuNzEgNy4wNGMuMzktLjM5LjM5LTEuMDIgMC0xLjQxbC0yLjM0LTIuMzRjLS4zOS0uMzktMS4wMi0uMzktMS40MSAwbC0xLjgzIDEuODMgMy43NSAzLjc1IDEuODMtMS44M3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-ellipses: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iNSIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxOSIgY3k9IjEyIiByPSIyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-error: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjE5IiByPSIyIi8+PHBhdGggZD0iTTEwIDNoNHYxMmgtNHoiLz48L2c+CjxwYXRoIGZpbGw9Im5vbmUiIGQ9Ik0wIDBoMjR2MjRIMHoiLz4KPC9zdmc+Cg==);
  --jp-icon-expand-all: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTggMmMxIDAgMTEgMCAxMiAwczIgMSAyIDJjMCAxIDAgMTEgMCAxMnMwIDItMiAyQzIwIDE0IDIwIDQgMjAgNFMxMCA0IDYgNGMwLTIgMS0yIDItMnoiIC8+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTE4IDhjMC0xLTEtMi0yLTJTNSA2IDQgNnMtMiAxLTIgMmMwIDEgMCAxMSAwIDEyczEgMiAyIDJjMSAwIDExIDAgMTIgMHMyLTEgMi0yYzAtMSAwLTExIDAtMTJ6bS0yIDB2MTJINFY4eiIgLz4KICAgICAgICA8cGF0aCBkPSJNMTEgMTBIOXYzSDZ2MmgzdjNoMnYtM2gzdi0yaC0zeiIgLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-extension: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwLjUgMTFIMTlWN2MwLTEuMS0uOS0yLTItMmgtNFYzLjVDMTMgMi4xMiAxMS44OCAxIDEwLjUgMVM4IDIuMTIgOCAzLjVWNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAydjMuOEgzLjVjMS40OSAwIDIuNyAxLjIxIDIuNyAyLjdzLTEuMjEgMi43LTIuNyAyLjdIMlYyMGMwIDEuMS45IDIgMiAyaDMuOHYtMS41YzAtMS40OSAxLjIxLTIuNyAyLjctMi43IDEuNDkgMCAyLjcgMS4yMSAyLjcgMi43VjIySDE3YzEuMSAwIDItLjkgMi0ydi00aDEuNWMxLjM4IDAgMi41LTEuMTIgMi41LTIuNVMyMS44OCAxMSAyMC41IDExeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-fast-forward: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTQgMThsOC41LTZMNCA2djEyem05LTEydjEybDguNS02TDEzIDZ6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-file-upload: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTZoNnYtNmg0bC03LTctNyA3aDR6bS00IDJoMTR2Mkg1eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-file: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuMyA4LjJsLTUuNS01LjVjLS4zLS4zLS43LS41LTEuMi0uNUgzLjljLS44LjEtMS42LjktMS42IDEuOHYxNC4xYzAgLjkuNyAxLjYgMS42IDEuNmgxNC4yYy45IDAgMS42LS43IDEuNi0xLjZWOS40Yy4xLS41LS4xLS45LS40LTEuMnptLTUuOC0zLjNsMy40IDMuNmgtMy40VjQuOXptMy45IDEyLjdINC43Yy0uMSAwLS4yIDAtLjItLjJWNC43YzAtLjIuMS0uMy4yLS4zaDcuMnY0LjRzMCAuOC4zIDEuMWMuMy4zIDEuMS4zIDEuMS4zaDQuM3Y3LjJzLS4xLjItLjIuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-filter-dot: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTE0LDEyVjE5Ljg4QzE0LjA0LDIwLjE4IDEzLjk0LDIwLjUgMTMuNzEsMjAuNzFDMTMuMzIsMjEuMSAxMi42OSwyMS4xIDEyLjMsMjAuNzFMMTAuMjksMTguN0MxMC4wNiwxOC40NyA5Ljk2LDE4LjE2IDEwLDE3Ljg3VjEySDkuOTdMNC4yMSw0LjYyQzMuODcsNC4xOSAzLjk1LDMuNTYgNC4zOCwzLjIyQzQuNTcsMy4wOCA0Ljc4LDMgNSwzVjNIMTlWM0MxOS4yMiwzIDE5LjQzLDMuMDggMTkuNjIsMy4yMkMyMC4wNSwzLjU2IDIwLjEzLDQuMTkgMTkuNzksNC42MkwxNC4wMywxMkgxNFoiIC8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWRvdCIgZmlsbD0iI0ZGRiI+CiAgICA8Y2lyY2xlIGN4PSIxOCIgY3k9IjE3IiByPSIzIj48L2NpcmNsZT4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-filter-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEwIDE4aDR2LTJoLTR2MnpNMyA2djJoMThWNkgzem0zIDdoMTJ2LTJINnYyeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-filter: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTE0LDEyVjE5Ljg4QzE0LjA0LDIwLjE4IDEzLjk0LDIwLjUgMTMuNzEsMjAuNzFDMTMuMzIsMjEuMSAxMi42OSwyMS4xIDEyLjMsMjAuNzFMMTAuMjksMTguN0MxMC4wNiwxOC40NyA5Ljk2LDE4LjE2IDEwLDE3Ljg3VjEySDkuOTdMNC4yMSw0LjYyQzMuODcsNC4xOSAzLjk1LDMuNTYgNC4zOCwzLjIyQzQuNTcsMy4wOCA0Ljc4LDMgNSwzVjNIMTlWM0MxOS4yMiwzIDE5LjQzLDMuMDggMTkuNjIsMy4yMkMyMC4wNSwzLjU2IDIwLjEzLDQuMTkgMTkuNzksNC42MkwxNC4wMywxMkgxNFoiIC8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-folder-favorite: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iIzAwMDAwMCI+CiAgPHBhdGggZD0iTTAgMGgyNHYyNEgwVjB6IiBmaWxsPSJub25lIi8+PHBhdGggY2xhc3M9ImpwLWljb24zIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxNjE2MSIgZD0iTTIwIDZoLThsLTItMkg0Yy0xLjEgMC0yIC45LTIgMnYxMmMwIDEuMS45IDIgMiAyaDE2YzEuMSAwIDItLjkgMi0yVjhjMC0xLjEtLjktMi0yLTJ6bS0yLjA2IDExTDE1IDE1LjI4IDEyLjA2IDE3bC43OC0zLjMzLTIuNTktMi4yNCAzLjQxLS4yOUwxNSA4bDEuMzQgMy4xNCAzLjQxLjI5LTIuNTkgMi4yNC43OCAzLjMzeiIvPgo8L3N2Zz4K);
  --jp-icon-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY4YzAtMS4xLS45LTItMi0yaC04bC0yLTJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-home: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iIzAwMDAwMCI+CiAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGNsYXNzPSJqcC1pY29uMyBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xMCAyMHYtNmg0djZoNXYtOGgzTDEyIDMgMiAxMmgzdjh6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-html5: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMDAiIGQ9Ik0xMDguNCAwaDIzdjIyLjhoMjEuMlYwaDIzdjY5aC0yM1Y0NmgtMjF2MjNoLTIzLjJNMjA2IDIzaC0yMC4zVjBoNjMuN3YyM0gyMjl2NDZoLTIzbTUzLjUtNjloMjQuMWwxNC44IDI0LjNMMzEzLjIgMGgyNC4xdjY5aC0yM1YzNC44bC0xNi4xIDI0LjgtMTYuMS0yNC44VjY5aC0yMi42bTg5LjItNjloMjN2NDYuMmgzMi42VjY5aC01NS42Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2U0NGQyNiIgZD0iTTEwNy42IDQ3MWwtMzMtMzcwLjRoMzYyLjhsLTMzIDM3MC4yTDI1NS43IDUxMiIvPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNmMTY1MjkiIGQ9Ik0yNTYgNDgwLjVWMTMxaDE0OC4zTDM3NiA0NDciLz4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNlYmViZWIiIGQ9Ik0xNDIgMTc2LjNoMTE0djQ1LjRoLTY0LjJsNC4yIDQ2LjVoNjB2NDUuM0gxNTQuNG0yIDIyLjhIMjAybDMuMiAzNi4zIDUwLjggMTMuNnY0Ny40bC05My4yLTI2Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIiBkPSJNMzY5LjYgMTc2LjNIMjU1Ljh2NDUuNGgxMDkuNm0tNC4xIDQ2LjVIMjU1Ljh2NDUuNGg1NmwtNS4zIDU5LTUwLjcgMTMuNnY0Ny4ybDkzLTI1LjgiLz4KPC9zdmc+Cg==);
  --jp-icon-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1icmFuZDQganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNGRkYiIGQ9Ik0yLjIgMi4yaDE3LjV2MTcuNUgyLjJ6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzNGNTFCNSIgZD0iTTIuMiAyLjJ2MTcuNWgxNy41bC4xLTE3LjVIMi4yem0xMi4xIDIuMmMxLjIgMCAyLjIgMSAyLjIgMi4ycy0xIDIuMi0yLjIgMi4yLTIuMi0xLTIuMi0yLjIgMS0yLjIgMi4yLTIuMnpNNC40IDE3LjZsMy4zLTguOCAzLjMgNi42IDIuMi0zLjIgNC40IDUuNEg0LjR6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-info: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUwLjk3OCA1MC45NzgiPgoJPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KCQk8cGF0aCBkPSJNNDMuNTIsNy40NThDMzguNzExLDIuNjQ4LDMyLjMwNywwLDI1LjQ4OSwwQzE4LjY3LDAsMTIuMjY2LDIuNjQ4LDcuNDU4LDcuNDU4CgkJCWMtOS45NDMsOS45NDEtOS45NDMsMjYuMTE5LDAsMzYuMDYyYzQuODA5LDQuODA5LDExLjIxMiw3LjQ1NiwxOC4wMzEsNy40NThjMCwwLDAuMDAxLDAsMC4wMDIsMAoJCQljNi44MTYsMCwxMy4yMjEtMi42NDgsMTguMDI5LTcuNDU4YzQuODA5LTQuODA5LDcuNDU3LTExLjIxMiw3LjQ1Ny0xOC4wM0M1MC45NzcsMTguNjcsNDguMzI4LDEyLjI2Niw0My41Miw3LjQ1OHoKCQkJIE00Mi4xMDYsNDIuMTA1Yy00LjQzMiw0LjQzMS0xMC4zMzIsNi44NzItMTYuNjE1LDYuODcyaC0wLjAwMmMtNi4yODUtMC4wMDEtMTIuMTg3LTIuNDQxLTE2LjYxNy02Ljg3MgoJCQljLTkuMTYyLTkuMTYzLTkuMTYyLTI0LjA3MSwwLTMzLjIzM0MxMy4zMDMsNC40NCwxOS4yMDQsMiwyNS40ODksMmM2LjI4NCwwLDEyLjE4NiwyLjQ0LDE2LjYxNyw2Ljg3MgoJCQljNC40MzEsNC40MzEsNi44NzEsMTAuMzMyLDYuODcxLDE2LjYxN0M0OC45NzcsMzEuNzcyLDQ2LjUzNiwzNy42NzUsNDIuMTA2LDQyLjEwNXoiLz4KCQk8cGF0aCBkPSJNMjMuNTc4LDMyLjIxOGMtMC4wMjMtMS43MzQsMC4xNDMtMy4wNTksMC40OTYtMy45NzJjMC4zNTMtMC45MTMsMS4xMS0xLjk5NywyLjI3Mi0zLjI1MwoJCQljMC40NjgtMC41MzYsMC45MjMtMS4wNjIsMS4zNjctMS41NzVjMC42MjYtMC43NTMsMS4xMDQtMS40NzgsMS40MzYtMi4xNzVjMC4zMzEtMC43MDcsMC40OTUtMS41NDEsMC40OTUtMi41CgkJCWMwLTEuMDk2LTAuMjYtMi4wODgtMC43NzktMi45NzljLTAuNTY1LTAuODc5LTEuNTAxLTEuMzM2LTIuODA2LTEuMzY5Yy0xLjgwMiwwLjA1Ny0yLjk4NSwwLjY2Ny0zLjU1LDEuODMyCgkJCWMtMC4zMDEsMC41MzUtMC41MDMsMS4xNDEtMC42MDcsMS44MTRjLTAuMTM5LDAuNzA3LTAuMjA3LDEuNDMyLTAuMjA3LDIuMTc0aC0yLjkzN2MtMC4wOTEtMi4yMDgsMC40MDctNC4xMTQsMS40OTMtNS43MTkKCQkJYzEuMDYyLTEuNjQsMi44NTUtMi40ODEsNS4zNzgtMi41MjdjMi4xNiwwLjAyMywzLjg3NCwwLjYwOCw1LjE0MSwxLjc1OGMxLjI3OCwxLjE2LDEuOTI5LDIuNzY0LDEuOTUsNC44MTEKCQkJYzAsMS4xNDItMC4xMzcsMi4xMTEtMC40MSwyLjkxMWMtMC4zMDksMC44NDUtMC43MzEsMS41OTMtMS4yNjgsMi4yNDNjLTAuNDkyLDAuNjUtMS4wNjgsMS4zMTgtMS43MywyLjAwMgoJCQljLTAuNjUsMC42OTctMS4zMTMsMS40NzktMS45ODcsMi4zNDZjLTAuMjM5LDAuMzc3LTAuNDI5LDAuNzc3LTAuNTY1LDEuMTk5Yy0wLjE2LDAuOTU5LTAuMjE3LDEuOTUxLTAuMTcxLDIuOTc5CgkJCUMyNi41ODksMzIuMjE4LDIzLjU3OCwzMi4yMTgsMjMuNTc4LDMyLjIxOHogTTIzLjU3OCwzOC4yMnYtMy40ODRoMy4wNzZ2My40ODRIMjMuNTc4eiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-inspector: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaW5zcGVjdG9yLWljb24tY29sb3IganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0tNSAxNEg0di00aDExdjR6bTAtNUg0VjloMTF2NHptNSA1aC00VjloNHY5eiIvPgo8L3N2Zz4K);
  --jp-icon-json: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtanNvbi1pY29uLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0Y5QTgyNSI+CiAgICA8cGF0aCBkPSJNMjAuMiAxMS44Yy0xLjYgMC0xLjcuNS0xLjcgMSAwIC40LjEuOS4xIDEuMy4xLjUuMS45LjEgMS4zIDAgMS43LTEuNCAyLjMtMy41IDIuM2gtLjl2LTEuOWguNWMxLjEgMCAxLjQgMCAxLjQtLjggMC0uMyAwLS42LS4xLTEgMC0uNC0uMS0uOC0uMS0xLjIgMC0xLjMgMC0xLjggMS4zLTItMS4zLS4yLTEuMy0uNy0xLjMtMiAwLS40LjEtLjguMS0xLjIuMS0uNC4xLS43LjEtMSAwLS44LS40LS43LTEuNC0uOGgtLjVWNC4xaC45YzIuMiAwIDMuNS43IDMuNSAyLjMgMCAuNC0uMS45LS4xIDEuMy0uMS41LS4xLjktLjEgMS4zIDAgLjUuMiAxIDEuNyAxdjEuOHpNMS44IDEwLjFjMS42IDAgMS43LS41IDEuNy0xIDAtLjQtLjEtLjktLjEtMS4zLS4xLS41LS4xLS45LS4xLTEuMyAwLTEuNiAxLjQtMi4zIDMuNS0yLjNoLjl2MS45aC0uNWMtMSAwLTEuNCAwLTEuNC44IDAgLjMgMCAuNi4xIDEgMCAuMi4xLjYuMSAxIDAgMS4zIDAgMS44LTEuMyAyQzYgMTEuMiA2IDExLjcgNiAxM2MwIC40LS4xLjgtLjEgMS4yLS4xLjMtLjEuNy0uMSAxIDAgLjguMy44IDEuNC44aC41djEuOWgtLjljLTIuMSAwLTMuNS0uNi0zLjUtMi4zIDAtLjQuMS0uOS4xLTEuMy4xLS41LjEtLjkuMS0xLjMgMC0uNS0uMi0xLTEuNy0xdi0xLjl6Ii8+CiAgICA8Y2lyY2xlIGN4PSIxMSIgY3k9IjEzLjgiIHI9IjIuMSIvPgogICAgPGNpcmNsZSBjeD0iMTEiIGN5PSI4LjIiIHI9IjIuMSIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-julia: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDMyNSAzMDAiPgogIDxnIGNsYXNzPSJqcC1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjY2IzYzMzIj4KICAgIDxwYXRoIGQ9Ik0gMTUwLjg5ODQzOCAyMjUgQyAxNTAuODk4NDM4IDI2Ni40MjE4NzUgMTE3LjMyMDMxMiAzMDAgNzUuODk4NDM4IDMwMCBDIDM0LjQ3NjU2MiAzMDAgMC44OTg0MzggMjY2LjQyMTg3NSAwLjg5ODQzOCAyMjUgQyAwLjg5ODQzOCAxODMuNTc4MTI1IDM0LjQ3NjU2MiAxNTAgNzUuODk4NDM4IDE1MCBDIDExNy4zMjAzMTIgMTUwIDE1MC44OTg0MzggMTgzLjU3ODEyNSAxNTAuODk4NDM4IDIyNSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzM4OTgyNiI+CiAgICA8cGF0aCBkPSJNIDIzNy41IDc1IEMgMjM3LjUgMTE2LjQyMTg3NSAyMDMuOTIxODc1IDE1MCAxNjIuNSAxNTAgQyAxMjEuMDc4MTI1IDE1MCA4Ny41IDExNi40MjE4NzUgODcuNSA3NSBDIDg3LjUgMzMuNTc4MTI1IDEyMS4wNzgxMjUgMCAxNjIuNSAwIEMgMjAzLjkyMTg3NSAwIDIzNy41IDMzLjU3ODEyNSAyMzcuNSA3NSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzk1NThiMiI+CiAgICA8cGF0aCBkPSJNIDMyNC4xMDE1NjIgMjI1IEMgMzI0LjEwMTU2MiAyNjYuNDIxODc1IDI5MC41MjM0MzggMzAwIDI0OS4xMDE1NjIgMzAwIEMgMjA3LjY3OTY4OCAzMDAgMTc0LjEwMTU2MiAyNjYuNDIxODc1IDE3NC4xMDE1NjIgMjI1IEMgMTc0LjEwMTU2MiAxODMuNTc4MTI1IDIwNy42Nzk2ODggMTUwIDI0OS4xMDE1NjIgMTUwIEMgMjkwLjUyMzQzOCAxNTAgMzI0LjEwMTU2MiAxODMuNTc4MTI1IDMyNC4xMDE1NjIgMjI1Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-jupyter-favicon: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUyIiBoZWlnaHQ9IjE2NSIgdmlld0JveD0iMCAwIDE1MiAxNjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgPGcgY2xhc3M9ImpwLWp1cHl0ZXItaWNvbi1jb2xvciIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA3ODk0NywgMTEwLjU4MjkyNykiIGQ9Ik03NS45NDIyODQyLDI5LjU4MDQ1NjEgQzQzLjMwMjM5NDcsMjkuNTgwNDU2MSAxNC43OTY3ODMyLDE3LjY1MzQ2MzQgMCwwIEM1LjUxMDgzMjExLDE1Ljg0MDY4MjkgMTUuNzgxNTM4OSwyOS41NjY3NzMyIDI5LjM5MDQ5NDcsMzkuMjc4NDE3MSBDNDIuOTk5Nyw0OC45ODk4NTM3IDU5LjI3MzcsNTQuMjA2NzgwNSA3NS45NjA1Nzg5LDU0LjIwNjc4MDUgQzkyLjY0NzQ1NzksNTQuMjA2NzgwNSAxMDguOTIxNDU4LDQ4Ljk4OTg1MzcgMTIyLjUzMDY2MywzOS4yNzg0MTcxIEMxMzYuMTM5NDUzLDI5LjU2Njc3MzIgMTQ2LjQxMDI4NCwxNS44NDA2ODI5IDE1MS45MjExNTgsMCBDMTM3LjA4Nzg2OCwxNy42NTM0NjM0IDEwOC41ODI1ODksMjkuNTgwNDU2MSA3NS45NDIyODQyLDI5LjU4MDQ1NjEgTDc1Ljk0MjI4NDIsMjkuNTgwNDU2MSBaIiAvPgogICAgPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMzczNjgsIDAuNzA0ODc4KSIgZD0iTTc1Ljk3ODQ1NzksMjQuNjI2NDA3MyBDMTA4LjYxODc2MywyNC42MjY0MDczIDEzNy4xMjQ0NTgsMzYuNTUzNDQxNSAxNTEuOTIxMTU4LDU0LjIwNjc4MDUgQzE0Ni40MTAyODQsMzguMzY2MjIyIDEzNi4xMzk0NTMsMjQuNjQwMTMxNyAxMjIuNTMwNjYzLDE0LjkyODQ4NzggQzEwOC45MjE0NTgsNS4yMTY4NDM5IDkyLjY0NzQ1NzksMCA3NS45NjA1Nzg5LDAgQzU5LjI3MzcsMCA0Mi45OTk3LDUuMjE2ODQzOSAyOS4zOTA0OTQ3LDE0LjkyODQ4NzggQzE1Ljc4MTUzODksMjQuNjQwMTMxNyA1LjUxMDgzMjExLDM4LjM2NjIyMiAwLDU0LjIwNjc4MDUgQzE0LjgzMzA4MTYsMzYuNTg5OTI5MyA0My4zMzg1Njg0LDI0LjYyNjQwNzMgNzUuOTc4NDU3OSwyNC42MjY0MDczIEw3NS45Nzg0NTc5LDI0LjYyNjQwNzMgWiIgLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzkiIGhlaWdodD0iNTEiIHZpZXdCb3g9IjAgMCAzOSA1MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTYzOCAtMjI4MSkiPgogICAgIDxnIGNsYXNzPSJqcC1qdXB5dGVyLWljb24tY29sb3IiIGZpbGw9IiNGMzc3MjYiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5Ljc0IDIzMTEuOTgpIiBkPSJNIDE4LjI2NDYgNy4xMzQxMUMgMTAuNDE0NSA3LjEzNDExIDMuNTU4NzIgNC4yNTc2IDAgMEMgMS4zMjUzOSAzLjgyMDQgMy43OTU1NiA3LjEzMDgxIDcuMDY4NiA5LjQ3MzAzQyAxMC4zNDE3IDExLjgxNTIgMTQuMjU1NyAxMy4wNzM0IDE4LjI2OSAxMy4wNzM0QyAyMi4yODIzIDEzLjA3MzQgMjYuMTk2MyAxMS44MTUyIDI5LjQ2OTQgOS40NzMwM0MgMzIuNzQyNCA3LjEzMDgxIDM1LjIxMjYgMy44MjA0IDM2LjUzOCAwQyAzMi45NzA1IDQuMjU3NiAyNi4xMTQ4IDcuMTM0MTEgMTguMjY0NiA3LjEzNDExWiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5LjczIDIyODUuNDgpIiBkPSJNIDE4LjI3MzMgNS45MzkzMUMgMjYuMTIzNSA1LjkzOTMxIDMyLjk3OTMgOC44MTU4MyAzNi41MzggMTMuMDczNEMgMzUuMjEyNiA5LjI1MzAzIDMyLjc0MjQgNS45NDI2MiAyOS40Njk0IDMuNjAwNEMgMjYuMTk2MyAxLjI1ODE4IDIyLjI4MjMgMCAxOC4yNjkgMEMgMTQuMjU1NyAwIDEwLjM0MTcgMS4yNTgxOCA3LjA2ODYgMy42MDA0QyAzLjc5NTU2IDUuOTQyNjIgMS4zMjUzOSA5LjI1MzAzIDAgMTMuMDczNEMgMy41Njc0NSA4LjgyNDYzIDEwLjQyMzIgNS45MzkzMSAxOC4yNzMzIDUuOTM5MzFaIi8+CiAgICA8L2c+CiAgICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjY5LjMgMjI4MS4zMSkiIGQ9Ik0gNS44OTM1MyAyLjg0NEMgNS45MTg4OSAzLjQzMTY1IDUuNzcwODUgNC4wMTM2NyA1LjQ2ODE1IDQuNTE2NDVDIDUuMTY1NDUgNS4wMTkyMiA0LjcyMTY4IDUuNDIwMTUgNC4xOTI5OSA1LjY2ODUxQyAzLjY2NDMgNS45MTY4OCAzLjA3NDQ0IDYuMDAxNTEgMi40OTgwNSA1LjkxMTcxQyAxLjkyMTY2IDUuODIxOSAxLjM4NDYzIDUuNTYxNyAwLjk1NDg5OCA1LjE2NDAxQyAwLjUyNTE3IDQuNzY2MzMgMC4yMjIwNTYgNC4yNDkwMyAwLjA4MzkwMzcgMy42Nzc1N0MgLTAuMDU0MjQ4MyAzLjEwNjExIC0wLjAyMTIzIDIuNTA2MTcgMC4xNzg3ODEgMS45NTM2NEMgMC4zNzg3OTMgMS40MDExIDAuNzM2ODA5IDAuOTIwODE3IDEuMjA3NTQgMC41NzM1MzhDIDEuNjc4MjYgMC4yMjYyNTkgMi4yNDA1NSAwLjAyNzU5MTkgMi44MjMyNiAwLjAwMjY3MjI5QyAzLjYwMzg5IC0wLjAzMDcxMTUgNC4zNjU3MyAwLjI0OTc4OSA0Ljk0MTQyIDAuNzgyNTUxQyA1LjUxNzExIDEuMzE1MzEgNS44NTk1NiAyLjA1Njc2IDUuODkzNTMgMi44NDRaIi8+CiAgICAgIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MzkuOCAyMzIzLjgxKSIgZD0iTSA3LjQyNzg5IDMuNTgzMzhDIDcuNDYwMDggNC4zMjQzIDcuMjczNTUgNS4wNTgxOSA2Ljg5MTkzIDUuNjkyMTNDIDYuNTEwMzEgNi4zMjYwNyA1Ljk1MDc1IDYuODMxNTYgNS4yODQxMSA3LjE0NDZDIDQuNjE3NDcgNy40NTc2MyAzLjg3MzcxIDcuNTY0MTQgMy4xNDcwMiA3LjQ1MDYzQyAyLjQyMDMyIDcuMzM3MTIgMS43NDMzNiA3LjAwODcgMS4yMDE4NCA2LjUwNjk1QyAwLjY2MDMyOCA2LjAwNTIgMC4yNzg2MSA1LjM1MjY4IDAuMTA1MDE3IDQuNjMyMDJDIC0wLjA2ODU3NTcgMy45MTEzNSAtMC4wMjYyMzYxIDMuMTU0OTQgMC4yMjY2NzUgMi40NTg1NkMgMC40Nzk1ODcgMS43NjIxNyAwLjkzMTY5NyAxLjE1NzEzIDEuNTI1NzYgMC43MjAwMzNDIDIuMTE5ODMgMC4yODI5MzUgMi44MjkxNCAwLjAzMzQzOTUgMy41NjM4OSAwLjAwMzEzMzQ0QyA0LjU0NjY3IC0wLjAzNzQwMzMgNS41MDUyOSAwLjMxNjcwNiA2LjIyOTYxIDAuOTg3ODM1QyA2Ljk1MzkzIDEuNjU4OTYgNy4zODQ4NCAyLjU5MjM1IDcuNDI3ODkgMy41ODMzOEwgNy40Mjc4OSAzLjU4MzM4WiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM4LjM2IDIyODYuMDYpIiBkPSJNIDIuMjc0NzEgNC4zOTYyOUMgMS44NDM2MyA0LjQxNTA4IDEuNDE2NzEgNC4zMDQ0NSAxLjA0Nzk5IDQuMDc4NDNDIDAuNjc5MjY4IDMuODUyNCAwLjM4NTMyOCAzLjUyMTE0IDAuMjAzMzcxIDMuMTI2NTZDIDAuMDIxNDEzNiAyLjczMTk4IC0wLjA0MDM3OTggMi4yOTE4MyAwLjAyNTgxMTYgMS44NjE4MUMgMC4wOTIwMDMxIDEuNDMxOCAwLjI4MzIwNCAxLjAzMTI2IDAuNTc1MjEzIDAuNzEwODgzQyAwLjg2NzIyMiAwLjM5MDUxIDEuMjQ2OTEgMC4xNjQ3MDggMS42NjYyMiAwLjA2MjA1OTJDIDIuMDg1NTMgLTAuMDQwNTg5NyAyLjUyNTYxIC0wLjAxNTQ3MTQgMi45MzA3NiAwLjEzNDIzNUMgMy4zMzU5MSAwLjI4Mzk0MSAzLjY4NzkyIDAuNTUxNTA1IDMuOTQyMjIgMC45MDMwNkMgNC4xOTY1MiAxLjI1NDYyIDQuMzQxNjkgMS42NzQzNiA0LjM1OTM1IDIuMTA5MTZDIDQuMzgyOTkgMi42OTEwNyA0LjE3Njc4IDMuMjU4NjkgMy43ODU5NyAzLjY4NzQ2QyAzLjM5NTE2IDQuMTE2MjQgMi44NTE2NiA0LjM3MTE2IDIuMjc0NzEgNC4zOTYyOUwgMi4yNzQ3MSA0LjM5NjI5WiIvPgogICAgPC9nPgogIDwvZz4+Cjwvc3ZnPgo=);
  --jp-icon-jupyterlab-wordmark: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIHZpZXdCb3g9IjAgMCAxODYwLjggNDc1Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0RTRFNEUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ4MC4xMzY0MDEsIDY0LjI3MTQ5MykiPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDU4Ljg3NTU2NikiPgogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA4NzYwMywgMC4xNDAyOTQpIj4KICAgICAgICA8cGF0aCBkPSJNLTQyNi45LDE2OS44YzAsNDguNy0zLjcsNjQuNy0xMy42LDc2LjRjLTEwLjgsMTAtMjUsMTUuNS0zOS43LDE1LjVsMy43LDI5IGMyMi44LDAuMyw0NC44LTcuOSw2MS45LTIzLjFjMTcuOC0xOC41LDI0LTQ0LjEsMjQtODMuM1YwSC00Mjd2MTcwLjFMLTQyNi45LDE2OS44TC00MjYuOSwxNjkuOHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTU1LjA0NTI5NiwgNTYuODM3MTA0KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTYyNDUzLCAxLjc5OTg0MikiPgogICAgICAgIDxwYXRoIGQ9Ik0tMzEyLDE0OGMwLDIxLDAsMzkuNSwxLjcsNTUuNGgtMzEuOGwtMi4xLTMzLjNoLTAuOGMtNi43LDExLjYtMTYuNCwyMS4zLTI4LDI3LjkgYy0xMS42LDYuNi0yNC44LDEwLTM4LjIsOS44Yy0zMS40LDAtNjktMTcuNy02OS04OVYwaDM2LjR2MTEyLjdjMCwzOC43LDExLjYsNjQuNyw0NC42LDY0LjdjMTAuMy0wLjIsMjAuNC0zLjUsMjguOS05LjQgYzguNS01LjksMTUuMS0xNC4zLDE4LjktMjMuOWMyLjItNi4xLDMuMy0xMi41LDMuMy0xOC45VjAuMmgzNi40VjE0OEgtMzEyTC0zMTIsMTQ4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTAuMDEzMzIyLCA1My40Nzk2MzgpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS43MDY0NTgsIDAuMjMxNDI1KSI+CiAgICAgICAgPHBhdGggZD0iTS00NzguNiw3MS40YzAtMjYtMC44LTQ3LTEuNy02Ni43aDMyLjdsMS43LDM0LjhoMC44YzcuMS0xMi41LDE3LjUtMjIuOCwzMC4xLTI5LjcgYzEyLjUtNywyNi43LTEwLjMsNDEtOS44YzQ4LjMsMCw4NC43LDQxLjcsODQuNywxMDMuM2MwLDczLjEtNDMuNywxMDkuMi05MSwxMDkuMmMtMTIuMSwwLjUtMjQuMi0yLjItMzUtNy44IGMtMTAuOC01LjYtMTkuOS0xMy45LTI2LjYtMjQuMmgtMC44VjI5MWgtMzZ2LTIyMEwtNDc4LjYsNzEuNEwtNDc4LjYsNzEuNHogTS00NDIuNiwxMjUuNmMwLjEsNS4xLDAuNiwxMC4xLDEuNywxNS4xIGMzLDEyLjMsOS45LDIzLjMsMTkuOCwzMS4xYzkuOSw3LjgsMjIuMSwxMi4xLDM0LjcsMTIuMWMzOC41LDAsNjAuNy0zMS45LDYwLjctNzguNWMwLTQwLjctMjEuMS03NS42LTU5LjUtNzUuNiBjLTEyLjksMC40LTI1LjMsNS4xLTM1LjMsMTMuNGMtOS45LDguMy0xNi45LDE5LjctMTkuNiwzMi40Yy0xLjUsNC45LTIuMywxMC0yLjUsMTUuMVYxMjUuNkwtNDQyLjYsMTI1LjZMLTQ0Mi42LDEyNS42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDYuNzQwNzI2LCA1Ni44MzcxMDQpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC43NTEyMjYsIDEuOTg5Mjk5KSI+CiAgICAgICAgPHBhdGggZD0iTS00NDAuOCwwbDQzLjcsMTIwLjFjNC41LDEzLjQsOS41LDI5LjQsMTIuOCw0MS43aDAuOGMzLjctMTIuMiw3LjktMjcuNywxMi44LTQyLjQgbDM5LjctMTE5LjJoMzguNUwtMzQ2LjksMTQ1Yy0yNiw2OS43LTQzLjcsMTA1LjQtNjguNiwxMjcuMmMtMTIuNSwxMS43LTI3LjksMjAtNDQuNiwyMy45bC05LjEtMzEuMSBjMTEuNy0zLjksMjIuNS0xMC4xLDMxLjgtMTguMWMxMy4yLTExLjEsMjMuNy0yNS4yLDMwLjYtNDEuMmMxLjUtMi44LDIuNS01LjcsMi45LTguOGMtMC4zLTMuMy0xLjItNi42LTIuNS05LjdMLTQ4MC4yLDAuMSBoMzkuN0wtNDQwLjgsMEwtNDQwLjgsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODIyLjc0ODEwNCwgMC4wMDAwMDApIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS40NjQwNTAsIDAuMzc4OTE0KSI+CiAgICAgICAgPHBhdGggZD0iTS00MTMuNywwdjU4LjNoNTJ2MjguMmgtNTJWMTk2YzAsMjUsNywzOS41LDI3LjMsMzkuNWM3LjEsMC4xLDE0LjItMC43LDIxLjEtMi41IGwxLjcsMjcuN2MtMTAuMywzLjctMjEuMyw1LjQtMzIuMiw1Yy03LjMsMC40LTE0LjYtMC43LTIxLjMtMy40Yy02LjgtMi43LTEyLjktNi44LTE3LjktMTIuMWMtMTAuMy0xMC45LTE0LjEtMjktMTQuMS01Mi45IFY4Ni41aC0zMVY1OC4zaDMxVjkuNkwtNDEzLjcsMEwtNDEzLjcsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTc0LjQzMzI4NiwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuOTkwMDM0LCAwLjYxMDMzOSkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDQ1LjgsMTEzYzAuOCw1MCwzMi4yLDcwLjYsNjguNiw3MC42YzE5LDAuNiwzNy45LTMsNTUuMy0xMC41bDYuMiwyNi40IGMtMjAuOSw4LjktNDMuNSwxMy4xLTY2LjIsMTIuNmMtNjEuNSwwLTk4LjMtNDEuMi05OC4zLTEwMi41Qy00ODAuMiw0OC4yLTQ0NC43LDAtMzg2LjUsMGM2NS4yLDAsODIuNyw1OC4zLDgyLjcsOTUuNyBjLTAuMSw1LjgtMC41LDExLjUtMS4yLDE3LjJoLTE0MC42SC00NDUuOEwtNDQ1LjgsMTEzeiBNLTMzOS4yLDg2LjZjMC40LTIzLjUtOS41LTYwLjEtNTAuNC02MC4xIGMtMzYuOCwwLTUyLjgsMzQuNC01NS43LDYwLjFILTMzOS4yTC0zMzkuMiw4Ni42TC0zMzkuMiw4Ni42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAxLjk2MTA1OCwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMTc5NjQwLCAwLjcwNTA2OCkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDc4LjYsNjhjMC0yMy45LTAuNC00NC41LTEuNy02My40aDMxLjhsMS4yLDM5LjloMS43YzkuMS0yNy4zLDMxLTQ0LjUsNTUuMy00NC41IGMzLjUtMC4xLDcsMC40LDEwLjMsMS4ydjM0LjhjLTQuMS0wLjktOC4yLTEuMy0xMi40LTEuMmMtMjUuNiwwLTQzLjcsMTkuNy00OC43LDQ3LjRjLTEsNS43LTEuNiwxMS41LTEuNywxNy4ydjEwOC4zaC0zNlY2OCBMLTQ3OC42LDY4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCBkPSJNMTM1Mi4zLDMyNi4yaDM3VjI4aC0zN1YzMjYuMnogTTE2MDQuOCwzMjYuMmMtMi41LTEzLjktMy40LTMxLjEtMy40LTQ4Ljd2LTc2IGMwLTQwLjctMTUuMS04My4xLTc3LjMtODMuMWMtMjUuNiwwLTUwLDcuMS02Ni44LDE4LjFsOC40LDI0LjRjMTQuMy05LjIsMzQtMTUuMSw1My0xNS4xYzQxLjYsMCw0Ni4yLDMwLjIsNDYuMiw0N3Y0LjIgYy03OC42LTAuNC0xMjIuMywyNi41LTEyMi4zLDc1LjZjMCwyOS40LDIxLDU4LjQsNjIuMiw1OC40YzI5LDAsNTAuOS0xNC4zLDYyLjItMzAuMmgxLjNsMi45LDI1LjZIMTYwNC44eiBNMTU2NS43LDI1Ny43IGMwLDMuOC0wLjgsOC0yLjEsMTEuOGMtNS45LDE3LjItMjIuNywzNC00OS4yLDM0Yy0xOC45LDAtMzQuOS0xMS4zLTM0LjktMzUuM2MwLTM5LjUsNDUuOC00Ni42LDg2LjItNDUuOFYyNTcuN3ogTTE2OTguNSwzMjYuMiBsMS43LTMzLjZoMS4zYzE1LjEsMjYuOSwzOC43LDM4LjIsNjguMSwzOC4yYzQ1LjQsMCw5MS4yLTM2LjEsOTEuMi0xMDguOGMwLjQtNjEuNy0zNS4zLTEwMy43LTg1LjctMTAzLjcgYy0zMi44LDAtNTYuMywxNC43LTY5LjMsMzcuNGgtMC44VjI4aC0zNi42djI0NS43YzAsMTguMS0wLjgsMzguNi0xLjcsNTIuNUgxNjk4LjV6IE0xNzA0LjgsMjA4LjJjMC01LjksMS4zLTEwLjksMi4xLTE1LjEgYzcuNi0yOC4xLDMxLjEtNDUuNCw1Ni4zLTQ1LjRjMzkuNSwwLDYwLjUsMzQuOSw2MC41LDc1LjZjMCw0Ni42LTIzLjEsNzguMS02MS44LDc4LjFjLTI2LjksMC00OC4zLTE3LjYtNTUuNS00My4zIGMtMC44LTQuMi0xLjctOC44LTEuNy0xMy40VjIwOC4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzYxNjE2MSIgZD0iTTE1IDlIOXY2aDZWOXptLTIgNGgtMnYtMmgydjJ6bTgtMlY5aC0yVjdjMC0xLjEtLjktMi0yLTJoLTJWM2gtMnYyaC0yVjNIOXYySDdjLTEuMSAwLTIgLjktMiAydjJIM3YyaDJ2MkgzdjJoMnYyYzAgMS4xLjkgMiAyIDJoMnYyaDJ2LTJoMnYyaDJ2LTJoMmMxLjEgMCAyLS45IDItMnYtMmgydi0yaC0ydi0yaDJ6bS00IDZIN1Y3aDEwdjEweiIvPgo8L3N2Zz4K);
  --jp-icon-keyboard: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMTdjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0tOSAzaDJ2MmgtMlY4em0wIDNoMnYyaC0ydi0yek04IDhoMnYySDhWOHptMCAzaDJ2Mkg4di0yem0tMSAySDV2LTJoMnYyem0wLTNINVY4aDJ2MnptOSA3SDh2LTJoOHYyem0wLTRoLTJ2LTJoMnYyem0wLTNoLTJWOGgydjJ6bTMgM2gtMnYtMmgydjJ6bTAtM2gtMlY4aDJ2MnoiLz4KPC9zdmc+Cg==);
  --jp-icon-launch: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMzIgMzIiIHdpZHRoPSIzMiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yNiwyOEg2YTIuMDAyNywyLjAwMjcsMCwwLDEtMi0yVjZBMi4wMDI3LDIuMDAyNywwLDAsMSw2LDRIMTZWNkg2VjI2SDI2VjE2aDJWMjZBMi4wMDI3LDIuMDAyNywwLDAsMSwyNiwyOFoiLz4KICAgIDxwb2x5Z29uIHBvaW50cz0iMjAgMiAyMCA0IDI2LjU4NiA0IDE4IDEyLjU4NiAxOS40MTQgMTQgMjggNS40MTQgMjggMTIgMzAgMTIgMzAgMiAyMCAyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-launcher: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkgMTlINVY1aDdWM0g1YTIgMiAwIDAwLTIgMnYxNGEyIDIgMCAwMDIgMmgxNGMxLjEgMCAyLS45IDItMnYtN2gtMnY3ek0xNCAzdjJoMy41OWwtOS44MyA5LjgzIDEuNDEgMS40MUwxOSA2LjQxVjEwaDJWM2gtN3oiLz4KPC9zdmc+Cg==);
  --jp-icon-line-form: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGZpbGw9IndoaXRlIiBkPSJNNS44OCA0LjEyTDEzLjc2IDEybC03Ljg4IDcuODhMOCAyMmwxMC0xMEw4IDJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-link: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMuOSAxMmMwLTEuNzEgMS4zOS0zLjEgMy4xLTMuMWg0VjdIN2MtMi43NiAwLTUgMi4yNC01IDVzMi4yNCA1IDUgNWg0di0xLjlIN2MtMS43MSAwLTMuMS0xLjM5LTMuMS0zLjF6TTggMTNoOHYtMkg4djJ6bTktNmgtNHYxLjloNGMxLjcxIDAgMy4xIDEuMzkgMy4xIDMuMXMtMS4zOSAzLjEtMy4xIDMuMWgtNFYxN2g0YzIuNzYgMCA1LTIuMjQgNS01cy0yLjI0LTUtNS01eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xOSA1djE0SDVWNWgxNG0xLjEtMkgzLjljLS41IDAtLjkuNC0uOS45djE2LjJjMCAuNC40LjkuOS45aDE2LjJjLjQgMCAuOS0uNS45LS45VjMuOWMwLS41LS41LS45LS45LS45ek0xMSA3aDZ2MmgtNlY3em0wIDRoNnYyaC02di0yem0wIDRoNnYyaC02ek03IDdoMnYySDd6bTAgNGgydjJIN3ptMCA0aDJ2Mkg3eiIvPgo8L3N2Zz4K);
  --jp-icon-markdown: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjN0IxRkEyIiBkPSJNNSAxNC45aDEybC02LjEgNnptOS40LTYuOGMwLTEuMy0uMS0yLjktLjEtNC41LS40IDEuNC0uOSAyLjktMS4zIDQuM2wtMS4zIDQuM2gtMkw4LjUgNy45Yy0uNC0xLjMtLjctMi45LTEtNC4zLS4xIDEuNi0uMSAzLjItLjIgNC42TDcgMTIuNEg0LjhsLjctMTFoMy4zTDEwIDVjLjQgMS4yLjcgMi43IDEgMy45LjMtMS4yLjctMi42IDEtMy45bDEuMi0zLjdoMy4zbC42IDExaC0yLjRsLS4zLTQuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-move-down: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNMTIuNDcxIDcuNTI4OTlDMTIuNzYzMiA3LjIzNjg0IDEyLjc2MzIgNi43NjMxNiAxMi40NzEgNi40NzEwMVY2LjQ3MTAxQzEyLjE3OSA2LjE3OTA1IDExLjcwNTcgNi4xNzg4NCAxMS40MTM1IDYuNDcwNTRMNy43NSAxMC4xMjc1VjEuNzVDNy43NSAxLjMzNTc5IDcuNDE0MjEgMSA3IDFWMUM2LjU4NTc5IDEgNi4yNSAxLjMzNTc5IDYuMjUgMS43NVYxMC4xMjc1TDIuNTk3MjYgNi40NjgyMkMyLjMwMzM4IDYuMTczODEgMS44MjY0MSA2LjE3MzU5IDEuNTMyMjYgNi40Njc3NFY2LjQ2Nzc0QzEuMjM4MyA2Ljc2MTcgMS4yMzgzIDcuMjM4MyAxLjUzMjI2IDcuNTMyMjZMNi4yOTI4OSAxMi4yOTI5QzYuNjgzNDIgMTIuNjgzNCA3LjMxNjU4IDEyLjY4MzQgNy43MDcxMSAxMi4yOTI5TDEyLjQ3MSA3LjUyODk5WiIgZmlsbD0iIzYxNjE2MSIvPgo8L3N2Zz4K);
  --jp-icon-move-up: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNMS41Mjg5OSA2LjQ3MTAxQzEuMjM2ODQgNi43NjMxNiAxLjIzNjg0IDcuMjM2ODQgMS41Mjg5OSA3LjUyODk5VjcuNTI4OTlDMS44MjA5NSA3LjgyMDk1IDIuMjk0MjYgNy44MjExNiAyLjU4NjQ5IDcuNTI5NDZMNi4yNSAzLjg3MjVWMTIuMjVDNi4yNSAxMi42NjQyIDYuNTg1NzkgMTMgNyAxM1YxM0M3LjQxNDIxIDEzIDcuNzUgMTIuNjY0MiA3Ljc1IDEyLjI1VjMuODcyNUwxMS40MDI3IDcuNTMxNzhDMTEuNjk2NiA3LjgyNjE5IDEyLjE3MzYgNy44MjY0MSAxMi40Njc3IDcuNTMyMjZWNy41MzIyNkMxMi43NjE3IDcuMjM4MyAxMi43NjE3IDYuNzYxNyAxMi40Njc3IDYuNDY3NzRMNy43MDcxMSAxLjcwNzExQzcuMzE2NTggMS4zMTY1OCA2LjY4MzQyIDEuMzE2NTggNi4yOTI4OSAxLjcwNzExTDEuNTI4OTkgNi40NzEwMVoiIGZpbGw9IiM2MTYxNjEiLz4KPC9zdmc+Cg==);
  --jp-icon-new-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDZoLThsLTItMkg0Yy0xLjExIDAtMS45OS44OS0xLjk5IDJMMiAxOGMwIDEuMTEuODkgMiAyIDJoMTZjMS4xMSAwIDItLjg5IDItMlY4YzAtMS4xMS0uODktMi0yLTJ6bS0xIDhoLTN2M2gtMnYtM2gtM3YtMmgzVjloMnYzaDN2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-not-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI1IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMTkgMTcuMTg0NCAyLjk2OTY4IDE0LjMwMzIgMS44NjA5NCAxMS40NDA5WiIvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24yIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOS4zMTU5MiA5LjMyMDMxKSIgZD0iTTcuMzY4NDIgMEwwIDcuMzY0NzkiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDkuMzE1OTIgMTYuNjgzNikgc2NhbGUoMSAtMSkiIGQ9Ik03LjM2ODQyIDBMMCA3LjM2NDc5Ii8+Cjwvc3ZnPgo=);
  --jp-icon-notebook: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtbm90ZWJvb2staWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNFRjZDMDAiPgogICAgPHBhdGggZD0iTTE4LjcgMy4zdjE1LjRIMy4zVjMuM2gxNS40bTEuNS0xLjVIMS44djE4LjNoMTguM2wuMS0xOC4zeiIvPgogICAgPHBhdGggZD0iTTE2LjUgMTYuNWwtNS40LTQuMy01LjYgNC4zdi0xMWgxMXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-numbering: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTQgMTlINlYxOS41SDVWMjAuNUg2VjIxSDRWMjJIN1YxOEg0VjE5Wk01IDEwSDZWNkg0VjdINVYxMFpNNCAxM0g1LjhMNCAxNS4xVjE2SDdWMTVINS4yTDcgMTIuOVYxMkg0VjEzWk05IDdWOUgyM1Y3SDlaTTkgMjFIMjNWMTlIOVYyMVpNOSAxNUgyM1YxM0g5VjE1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-offline-bolt: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDIuMDJjLTUuNTEgMC05Ljk4IDQuNDctOS45OCA5Ljk4czQuNDcgOS45OCA5Ljk4IDkuOTggOS45OC00LjQ3IDkuOTgtOS45OFMxNy41MSAyLjAyIDEyIDIuMDJ6TTExLjQ4IDIwdi02LjI2SDhMMTMgNHY2LjI2aDMuMzVMMTEuNDggMjB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-palette: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE4IDEzVjIwSDRWNkg5LjAyQzkuMDcgNS4yOSA5LjI0IDQuNjIgOS41IDRINEMyLjkgNCAyIDQuOSAyIDZWMjBDMiAyMS4xIDIuOSAyMiA0IDIySDE4QzE5LjEgMjIgMjAgMjEuMSAyMCAyMFYxNUwxOCAxM1pNMTkuMyA4Ljg5QzE5Ljc0IDguMTkgMjAgNy4zOCAyMCA2LjVDMjAgNC4wMSAxNy45OSAyIDE1LjUgMkMxMy4wMSAyIDExIDQuMDEgMTEgNi41QzExIDguOTkgMTMuMDEgMTEgMTUuNDkgMTFDMTYuMzcgMTEgMTcuMTkgMTAuNzQgMTcuODggMTAuM0wyMSAxMy40MkwyMi40MiAxMkwxOS4zIDguODlaTTE1LjUgOUMxNC4xMiA5IDEzIDcuODggMTMgNi41QzEzIDUuMTIgMTQuMTIgNCAxNS41IDRDMTYuODggNCAxOCA1LjEyIDE4IDYuNUMxOCA3Ljg4IDE2Ljg4IDkgMTUuNSA5WiIvPgogICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00IDZIOS4wMTg5NEM5LjAwNjM5IDYuMTY1MDIgOSA2LjMzMTc2IDkgNi41QzkgOC44MTU3NyAxMC4yMTEgMTAuODQ4NyAxMi4wMzQzIDEySDlWMTRIMTZWMTIuOTgxMUMxNi41NzAzIDEyLjkzNzcgMTcuMTIgMTIuODIwNyAxNy42Mzk2IDEyLjYzOTZMMTggMTNWMjBINFY2Wk04IDhINlYxMEg4VjhaTTYgMTJIOFYxNEg2VjEyWk04IDE2SDZWMThIOFYxNlpNOSAxNkgxNlYxOEg5VjE2WiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-paste: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE5IDJoLTQuMThDMTQuNC44NCAxMy4zIDAgMTIgMGMtMS4zIDAtMi40Ljg0LTIuODIgMkg1Yy0xLjEgMC0yIC45LTIgMnYxNmMwIDEuMS45IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjRjMC0xLjEtLjktMi0yLTJ6bS03IDBjLjU1IDAgMSAuNDUgMSAxcy0uNDUgMS0xIDEtMS0uNDUtMS0xIC40NS0xIDEtMXptNyAxOEg1VjRoMnYzaDEwVjRoMnYxNnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-pdf: url(data:image/svg+xml;base64,PHN2ZwogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMiAyMiIgd2lkdGg9IjE2Ij4KICAgIDxwYXRoIHRyYW5zZm9ybT0icm90YXRlKDQ1KSIgY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0ZGMkEyQSIKICAgICAgIGQ9Im0gMjIuMzQ0MzY5LC0zLjAxNjM2NDIgaCA1LjYzODYwNCB2IDEuNTc5MjQzMyBoIC0zLjU0OTIyNyB2IDEuNTA4NjkyOTkgaCAzLjMzNzU3NiBWIDEuNjUwODE1NCBoIC0zLjMzNzU3NiB2IDMuNDM1MjYxMyBoIC0yLjA4OTM3NyB6IG0gLTcuMTM2NDQ0LDEuNTc5MjQzMyB2IDQuOTQzOTU0MyBoIDAuNzQ4OTIgcSAxLjI4MDc2MSwwIDEuOTUzNzAzLC0wLjYzNDk1MzUgMC42NzgzNjksLTAuNjM0OTUzNSAwLjY3ODM2OSwtMS44NDUxNjQxIDAsLTEuMjA0NzgzNTUgLTAuNjcyOTQyLC0xLjgzNDMxMDExIC0wLjY3Mjk0MiwtMC42Mjk1MjY1OSAtMS45NTkxMywtMC42Mjk1MjY1OSB6IG0gLTIuMDg5Mzc3LC0xLjU3OTI0MzMgaCAyLjIwMzM0MyBxIDEuODQ1MTY0LDAgMi43NDYwMzksMC4yNjU5MjA3IDAuOTA2MzAxLDAuMjYwNDkzNyAxLjU1MjEwOCwwLjg5MDAyMDMgMC41Njk4MywwLjU0ODEyMjMgMC44NDY2MDUsMS4yNjQ0ODAwNiAwLjI3Njc3NCwwLjcxNjM1NzgxIDAuMjc2Nzc0LDEuNjIyNjU4OTQgMCwwLjkxNzE1NTEgLTAuMjc2Nzc0LDEuNjM4OTM5OSAtMC4yNzY3NzUsMC43MTYzNTc4IC0wLjg0NjYwNSwxLjI2NDQ4IC0wLjY1MTIzNCwwLjYyOTUyNjYgLTEuNTYyOTYyLDAuODk1NDQ3MyAtMC45MTE3MjgsMC4yNjA0OTM3IC0yLjczNTE4NSwwLjI2MDQ5MzcgaCAtMi4yMDMzNDMgeiBtIC04LjE0NTg1NjUsMCBoIDMuNDY3ODIzIHEgMS41NDY2ODE2LDAgMi4zNzE1Nzg1LDAuNjg5MjIzIDAuODMwMzI0LDAuNjgzNzk2MSAwLjgzMDMyNCwxLjk1MzcwMzE0IDAsMS4yNzUzMzM5NyAtMC44MzAzMjQsMS45NjQ1NTcwNiBRIDkuOTg3MTk2MSwyLjI3NDkxNSA4LjQ0MDUxNDUsMi4yNzQ5MTUgSCA3LjA2MjA2ODQgViA1LjA4NjA3NjcgSCA0Ljk3MjY5MTUgWiBtIDIuMDg5Mzc2OSwxLjUxNDExOTkgdiAyLjI2MzAzOTQzIGggMS4xNTU5NDEgcSAwLjYwNzgxODgsMCAwLjkzODg2MjksLTAuMjkzMDU1NDcgMC4zMzEwNDQxLC0wLjI5ODQ4MjQxIDAuMzMxMDQ0MSwtMC44NDExNzc3MiAwLC0wLjU0MjY5NTMxIC0wLjMzMTA0NDEsLTAuODM1NzUwNzQgLTAuMzMxMDQ0MSwtMC4yOTMwNTU1IC0wLjkzODg2MjksLTAuMjkzMDU1NSB6IgovPgo8L3N2Zz4K);
  --jp-icon-python: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iLTEwIC0xMCAxMzEuMTYxMzYxNjk0MzM1OTQgMTMyLjM4ODk5OTkzODk2NDg0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMzA2OTk4IiBkPSJNIDU0LjkxODc4NSw5LjE5Mjc0MjFlLTQgQyA1MC4zMzUxMzIsMC4wMjIyMTcyNyA0NS45NTc4NDYsMC40MTMxMzY5NyA0Mi4xMDYyODUsMS4wOTQ2NjkzIDMwLjc2MDA2OSwzLjA5OTE3MzEgMjguNzAwMDM2LDcuMjk0NzcxNCAyOC43MDAwMzUsMTUuMDMyMTY5IHYgMTAuMjE4NzUgaCAyNi44MTI1IHYgMy40MDYyNSBoIC0yNi44MTI1IC0xMC4wNjI1IGMgLTcuNzkyNDU5LDAgLTE0LjYxNTc1ODgsNC42ODM3MTcgLTE2Ljc0OTk5OTgsMTMuNTkzNzUgLTIuNDYxODE5OTgsMTAuMjEyOTY2IC0yLjU3MTAxNTA4LDE2LjU4NjAyMyAwLDI3LjI1IDEuOTA1OTI4Myw3LjkzNzg1MiA2LjQ1NzU0MzIsMTMuNTkzNzQ4IDE0LjI0OTk5OTgsMTMuNTkzNzUgaCA5LjIxODc1IHYgLTEyLjI1IGMgMCwtOC44NDk5MDIgNy42NTcxNDQsLTE2LjY1NjI0OCAxNi43NSwtMTYuNjU2MjUgaCAyNi43ODEyNSBjIDcuNDU0OTUxLDAgMTMuNDA2MjUzLC02LjEzODE2NCAxMy40MDYyNSwtMTMuNjI1IHYgLTI1LjUzMTI1IGMgMCwtNy4yNjYzMzg2IC02LjEyOTk4LC0xMi43MjQ3NzcxIC0xMy40MDYyNSwtMTMuOTM3NDk5NyBDIDY0LjI4MTU0OCwwLjMyNzk0Mzk3IDU5LjUwMjQzOCwtMC4wMjAzNzkwMyA1NC45MTg3ODUsOS4xOTI3NDIxZS00IFogbSAtMTQuNSw4LjIxODc1MDEyNTc5IGMgMi43Njk1NDcsMCA1LjAzMTI1LDIuMjk4NjQ1NiA1LjAzMTI1LDUuMTI0OTk5NiAtMmUtNiwyLjgxNjMzNiAtMi4yNjE3MDMsNS4wOTM3NSAtNS4wMzEyNSw1LjA5Mzc1IC0yLjc3OTQ3NiwtMWUtNiAtNS4wMzEyNSwtMi4yNzc0MTUgLTUuMDMxMjUsLTUuMDkzNzUgLTEwZS03LC0yLjgyNjM1MyAyLjI1MTc3NCwtNS4xMjQ5OTk2IDUuMDMxMjUsLTUuMTI0OTk5NiB6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2ZmZDQzYiIgZD0ibSA4NS42Mzc1MzUsMjguNjU3MTY5IHYgMTEuOTA2MjUgYyAwLDkuMjMwNzU1IC03LjgyNTg5NSwxNi45OTk5OTkgLTE2Ljc1LDE3IGggLTI2Ljc4MTI1IGMgLTcuMzM1ODMzLDAgLTEzLjQwNjI0OSw2LjI3ODQ4MyAtMTMuNDA2MjUsMTMuNjI1IHYgMjUuNTMxMjQ3IGMgMCw3LjI2NjM0NCA2LjMxODU4OCwxMS41NDAzMjQgMTMuNDA2MjUsMTMuNjI1MDA0IDguNDg3MzMxLDIuNDk1NjEgMTYuNjI2MjM3LDIuOTQ2NjMgMjYuNzgxMjUsMCA2Ljc1MDE1NSwtMS45NTQzOSAxMy40MDYyNTMsLTUuODg3NjEgMTMuNDA2MjUsLTEzLjYyNTAwNCBWIDg2LjUwMDkxOSBoIC0yNi43ODEyNSB2IC0zLjQwNjI1IGggMjYuNzgxMjUgMTMuNDA2MjU0IGMgNy43OTI0NjEsMCAxMC42OTYyNTEsLTUuNDM1NDA4IDEzLjQwNjI0MSwtMTMuNTkzNzUgMi43OTkzMywtOC4zOTg4ODYgMi42ODAyMiwtMTYuNDc1Nzc2IDAsLTI3LjI1IC0xLjkyNTc4LC03Ljc1NzQ0MSAtNS42MDM4NywtMTMuNTkzNzUgLTEzLjQwNjI0MSwtMTMuNTkzNzUgeiBtIC0xNS4wNjI1LDY0LjY1NjI1IGMgMi43Nzk0NzgsM2UtNiA1LjAzMTI1LDIuMjc3NDE3IDUuMDMxMjUsNS4wOTM3NDcgLTJlLTYsMi44MjYzNTQgLTIuMjUxNzc1LDUuMTI1MDA0IC01LjAzMTI1LDUuMTI1MDA0IC0yLjc2OTU1LDAgLTUuMDMxMjUsLTIuMjk4NjUgLTUuMDMxMjUsLTUuMTI1MDA0IDJlLTYsLTIuODE2MzMgMi4yNjE2OTcsLTUuMDkzNzQ3IDUuMDMxMjUsLTUuMDkzNzQ3IHoiLz4KPC9zdmc+Cg==);
  --jp-icon-r-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjE5NkYzIiBkPSJNNC40IDIuNWMxLjItLjEgMi45LS4zIDQuOS0uMyAyLjUgMCA0LjEuNCA1LjIgMS4zIDEgLjcgMS41IDEuOSAxLjUgMy41IDAgMi0xLjQgMy41LTIuOSA0LjEgMS4yLjQgMS43IDEuNiAyLjIgMyAuNiAxLjkgMSAzLjkgMS4zIDQuNmgtMy44Yy0uMy0uNC0uOC0xLjctMS4yLTMuN3MtMS4yLTIuNi0yLjYtMi42aC0uOXY2LjRINC40VjIuNXptMy43IDYuOWgxLjRjMS45IDAgMi45LS45IDIuOS0yLjNzLTEtMi4zLTIuOC0yLjNjLS43IDAtMS4zIDAtMS42LjJ2NC41aC4xdi0uMXoiLz4KPC9zdmc+Cg==);
  --jp-icon-react: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMTUwIDE1MCA1NDEuOSAyOTUuMyI+CiAgPGcgY2xhc3M9ImpwLWljb24tYnJhbmQyIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxREFGQiI+CiAgICA8cGF0aCBkPSJNNjY2LjMgMjk2LjVjMC0zMi41LTQwLjctNjMuMy0xMDMuMS04Mi40IDE0LjQtNjMuNiA4LTExNC4yLTIwLjItMTMwLjQtNi41LTMuOC0xNC4xLTUuNi0yMi40LTUuNnYyMi4zYzQuNiAwIDguMy45IDExLjQgMi42IDEzLjYgNy44IDE5LjUgMzcuNSAxNC45IDc1LjctMS4xIDkuNC0yLjkgMTkuMy01LjEgMjkuNC0xOS42LTQuOC00MS04LjUtNjMuNS0xMC45LTEzLjUtMTguNS0yNy41LTM1LjMtNDEuNi01MCAzMi42LTMwLjMgNjMuMi00Ni45IDg0LTQ2LjlWNzhjLTI3LjUgMC02My41IDE5LjYtOTkuOSA1My42LTM2LjQtMzMuOC03Mi40LTUzLjItOTkuOS01My4ydjIyLjNjMjAuNyAwIDUxLjQgMTYuNSA4NCA0Ni42LTE0IDE0LjctMjggMzEuNC00MS4zIDQ5LjktMjIuNiAyLjQtNDQgNi4xLTYzLjYgMTEtMi4zLTEwLTQtMTkuNy01LjItMjktNC43LTM4LjIgMS4xLTY3LjkgMTQuNi03NS44IDMtMS44IDYuOS0yLjYgMTEuNS0yLjZWNzguNWMtOC40IDAtMTYgMS44LTIyLjYgNS42LTI4LjEgMTYuMi0zNC40IDY2LjctMTkuOSAxMzAuMS02Mi4yIDE5LjItMTAyLjcgNDkuOS0xMDIuNyA4Mi4zIDAgMzIuNSA0MC43IDYzLjMgMTAzLjEgODIuNC0xNC40IDYzLjYtOCAxMTQuMiAyMC4yIDEzMC40IDYuNSAzLjggMTQuMSA1LjYgMjIuNSA1LjYgMjcuNSAwIDYzLjUtMTkuNiA5OS45LTUzLjYgMzYuNCAzMy44IDcyLjQgNTMuMiA5OS45IDUzLjIgOC40IDAgMTYtMS44IDIyLjYtNS42IDI4LjEtMTYuMiAzNC40LTY2LjcgMTkuOS0xMzAuMSA2Mi0xOS4xIDEwMi41LTQ5LjkgMTAyLjUtODIuM3ptLTEzMC4yLTY2LjdjLTMuNyAxMi45LTguMyAyNi4yLTEzLjUgMzkuNS00LjEtOC04LjQtMTYtMTMuMS0yNC00LjYtOC05LjUtMTUuOC0xNC40LTIzLjQgMTQuMiAyLjEgMjcuOSA0LjcgNDEgNy45em0tNDUuOCAxMDYuNWMtNy44IDEzLjUtMTUuOCAyNi4zLTI0LjEgMzguMi0xNC45IDEuMy0zMCAyLTQ1LjIgMi0xNS4xIDAtMzAuMi0uNy00NS0xLjktOC4zLTExLjktMTYuNC0yNC42LTI0LjItMzgtNy42LTEzLjEtMTQuNS0yNi40LTIwLjgtMzkuOCA2LjItMTMuNCAxMy4yLTI2LjggMjAuNy0zOS45IDcuOC0xMy41IDE1LjgtMjYuMyAyNC4xLTM4LjIgMTQuOS0xLjMgMzAtMiA0NS4yLTIgMTUuMSAwIDMwLjIuNyA0NSAxLjkgOC4zIDExLjkgMTYuNCAyNC42IDI0LjIgMzggNy42IDEzLjEgMTQuNSAyNi40IDIwLjggMzkuOC02LjMgMTMuNC0xMy4yIDI2LjgtMjAuNyAzOS45em0zMi4zLTEzYzUuNCAxMy40IDEwIDI2LjggMTMuOCAzOS44LTEzLjEgMy4yLTI2LjkgNS45LTQxLjIgOCA0LjktNy43IDkuOC0xNS42IDE0LjQtMjMuNyA0LjYtOCA4LjktMTYuMSAxMy0yNC4xek00MjEuMiA0MzBjLTkuMy05LjYtMTguNi0yMC4zLTI3LjgtMzIgOSAuNCAxOC4yLjcgMjcuNS43IDkuNCAwIDE4LjctLjIgMjcuOC0uNy05IDExLjctMTguMyAyMi40LTI3LjUgMzJ6bS03NC40LTU4LjljLTE0LjItMi4xLTI3LjktNC43LTQxLTcuOSAzLjctMTIuOSA4LjMtMjYuMiAxMy41LTM5LjUgNC4xIDggOC40IDE2IDEzLjEgMjQgNC43IDggOS41IDE1LjggMTQuNCAyMy40ek00MjAuNyAxNjNjOS4zIDkuNiAxOC42IDIwLjMgMjcuOCAzMi05LS40LTE4LjItLjctMjcuNS0uNy05LjQgMC0xOC43LjItMjcuOC43IDktMTEuNyAxOC4zLTIyLjQgMjcuNS0zMnptLTc0IDU4LjljLTQuOSA3LjctOS44IDE1LjYtMTQuNCAyMy43LTQuNiA4LTguOSAxNi0xMyAyNC01LjQtMTMuNC0xMC0yNi44LTEzLjgtMzkuOCAxMy4xLTMuMSAyNi45LTUuOCA0MS4yLTcuOXptLTkwLjUgMTI1LjJjLTM1LjQtMTUuMS01OC4zLTM0LjktNTguMy01MC42IDAtMTUuNyAyMi45LTM1LjYgNTguMy01MC42IDguNi0zLjcgMTgtNyAyNy43LTEwLjEgNS43IDE5LjYgMTMuMiA0MCAyMi41IDYwLjktOS4yIDIwLjgtMTYuNiA0MS4xLTIyLjIgNjAuNi05LjktMy4xLTE5LjMtNi41LTI4LTEwLjJ6TTMxMCA0OTBjLTEzLjYtNy44LTE5LjUtMzcuNS0xNC45LTc1LjcgMS4xLTkuNCAyLjktMTkuMyA1LjEtMjkuNCAxOS42IDQuOCA0MSA4LjUgNjMuNSAxMC45IDEzLjUgMTguNSAyNy41IDM1LjMgNDEuNiA1MC0zMi42IDMwLjMtNjMuMiA0Ni45LTg0IDQ2LjktNC41LS4xLTguMy0xLTExLjMtMi43em0yMzcuMi03Ni4yYzQuNyAzOC4yLTEuMSA2Ny45LTE0LjYgNzUuOC0zIDEuOC02LjkgMi42LTExLjUgMi42LTIwLjcgMC01MS40LTE2LjUtODQtNDYuNiAxNC0xNC43IDI4LTMxLjQgNDEuMy00OS45IDIyLjYtMi40IDQ0LTYuMSA2My42LTExIDIuMyAxMC4xIDQuMSAxOS44IDUuMiAyOS4xem0zOC41LTY2LjdjLTguNiAzLjctMTggNy0yNy43IDEwLjEtNS43LTE5LjYtMTMuMi00MC0yMi41LTYwLjkgOS4yLTIwLjggMTYuNi00MS4xIDIyLjItNjAuNiA5LjkgMy4xIDE5LjMgNi41IDI4LjEgMTAuMiAzNS40IDE1LjEgNTguMyAzNC45IDU4LjMgNTAuNi0uMSAxNS43LTIzIDM1LjYtNTguNCA1MC42ek0zMjAuOCA3OC40eiIvPgogICAgPGNpcmNsZSBjeD0iNDIwLjkiIGN5PSIyOTYuNSIgcj0iNDUuNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-redo: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTE4LjQgMTAuNkMxNi41NSA4Ljk5IDE0LjE1IDggMTEuNSA4Yy00LjY1IDAtOC41OCAzLjAzLTkuOTYgNy4yMkwzLjkgMTZjMS4wNS0zLjE5IDQuMDUtNS41IDcuNi01LjUgMS45NSAwIDMuNzMuNzIgNS4xMiAxLjg4TDEzIDE2aDlWN2wtMy42IDMuNnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-refresh: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTkgMTMuNWMtMi40OSAwLTQuNS0yLjAxLTQuNS00LjVTNi41MSA0LjUgOSA0LjVjMS4yNCAwIDIuMzYuNTIgMy4xNyAxLjMzTDEwIDhoNVYzbC0xLjc2IDEuNzZDMTIuMTUgMy42OCAxMC42NiAzIDkgMyA1LjY5IDMgMy4wMSA1LjY5IDMuMDEgOVM1LjY5IDE1IDkgMTVjMi45NyAwIDUuNDMtMi4xNiA1LjktNWgtMS41MmMtLjQ2IDItMi4yNCAzLjUtNC4zOCAzLjV6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-regex: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiBmaWxsPSIjRkZGIj4KICAgIDxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjUuNSIgY3k9IjE0LjUiIHI9IjEuNSIvPgogICAgPHJlY3QgeD0iMTIiIHk9IjQiIGNsYXNzPSJzdDIiIHdpZHRoPSIxIiBoZWlnaHQ9IjgiLz4KICAgIDxyZWN0IHg9IjguNSIgeT0iNy41IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjg2NiAtMC41IDAuNSAwLjg2NiAtMi4zMjU1IDcuMzIxOSkiIGNsYXNzPSJzdDIiIHdpZHRoPSI4IiBoZWlnaHQ9IjEiLz4KICAgIDxyZWN0IHg9IjEyIiB5PSI0IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjUgLTAuODY2IDAuODY2IDAuNSAtMC42Nzc5IDE0LjgyNTIpIiBjbGFzcz0ic3QyIiB3aWR0aD0iMSIgaGVpZ2h0PSI4Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-run: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTggNXYxNGwxMS03eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-running: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMjU2IDhDMTE5IDggOCAxMTkgOCAyNTZzMTExIDI0OCAyNDggMjQ4IDI0OC0xMTEgMjQ4LTI0OFMzOTMgOCAyNTYgOHptOTYgMzI4YzAgOC44LTcuMiAxNi0xNiAxNkgxNzZjLTguOCAwLTE2LTcuMi0xNi0xNlYxNzZjMC04LjggNy4yLTE2IDE2LTE2aDE2MGM4LjggMCAxNiA3LjIgMTYgMTZ2MTYweiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-save: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE3IDNINWMtMS4xMSAwLTIgLjktMiAydjE0YzAgMS4xLjg5IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjdsLTQtNHptLTUgMTZjLTEuNjYgMC0zLTEuMzQtMy0zczEuMzQtMyAzLTMgMyAxLjM0IDMgMy0xLjM0IDMtMyAzem0zLTEwSDVWNWgxMHY0eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-search: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-settings: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuNDMgMTIuOThjLjA0LS4zMi4wNy0uNjQuMDctLjk4cy0uMDMtLjY2LS4wNy0uOThsMi4xMS0xLjY1Yy4xOS0uMTUuMjQtLjQyLjEyLS42NGwtMi0zLjQ2Yy0uMTItLjIyLS4zOS0uMy0uNjEtLjIybC0yLjQ5IDFjLS41Mi0uNC0xLjA4LS43My0xLjY5LS45OGwtLjM4LTIuNjVBLjQ4OC40ODggMCAwMDE0IDJoLTRjLS4yNSAwLS40Ni4xOC0uNDkuNDJsLS4zOCAyLjY1Yy0uNjEuMjUtMS4xNy41OS0xLjY5Ljk4bC0yLjQ5LTFjLS4yMy0uMDktLjQ5IDAtLjYxLjIybC0yIDMuNDZjLS4xMy4yMi0uMDcuNDkuMTIuNjRsMi4xMSAxLjY1Yy0uMDQuMzItLjA3LjY1LS4wNy45OHMuMDMuNjYuMDcuOThsLTIuMTEgMS42NWMtLjE5LjE1LS4yNC40Mi0uMTIuNjRsMiAzLjQ2Yy4xMi4yMi4zOS4zLjYxLjIybDIuNDktMWMuNTIuNCAxLjA4LjczIDEuNjkuOThsLjM4IDIuNjVjLjAzLjI0LjI0LjQyLjQ5LjQyaDRjLjI1IDAgLjQ2LS4xOC40OS0uNDJsLjM4LTIuNjVjLjYxLS4yNSAxLjE3LS41OSAxLjY5LS45OGwyLjQ5IDFjLjIzLjA5LjQ5IDAgLjYxLS4yMmwyLTMuNDZjLjEyLS4yMi4wNy0uNDktLjEyLS42NGwtMi4xMS0xLjY1ek0xMiAxNS41Yy0xLjkzIDAtMy41LTEuNTctMy41LTMuNXMxLjU3LTMuNSAzLjUtMy41IDMuNSAxLjU3IDMuNSAzLjUtMS41NyAzLjUtMy41IDMuNXoiLz4KPC9zdmc+Cg==);
  --jp-icon-share: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTSAxOCAyIEMgMTYuMzU0OTkgMiAxNSAzLjM1NDk5MDQgMTUgNSBDIDE1IDUuMTkwOTUyOSAxNS4wMjE3OTEgNS4zNzcxMjI0IDE1LjA1NjY0MSA1LjU1ODU5MzggTCA3LjkyMTg3NSA5LjcyMDcwMzEgQyA3LjM5ODUzOTkgOS4yNzc4NTM5IDYuNzMyMDc3MSA5IDYgOSBDIDQuMzU0OTkwNCA5IDMgMTAuMzU0OTkgMyAxMiBDIDMgMTMuNjQ1MDEgNC4zNTQ5OTA0IDE1IDYgMTUgQyA2LjczMjA3NzEgMTUgNy4zOTg1Mzk5IDE0LjcyMjE0NiA3LjkyMTg3NSAxNC4yNzkyOTcgTCAxNS4wNTY2NDEgMTguNDM5NDUzIEMgMTUuMDIxNTU1IDE4LjYyMTUxNCAxNSAxOC44MDgzODYgMTUgMTkgQyAxNSAyMC42NDUwMSAxNi4zNTQ5OSAyMiAxOCAyMiBDIDE5LjY0NTAxIDIyIDIxIDIwLjY0NTAxIDIxIDE5IEMgMjEgMTcuMzU0OTkgMTkuNjQ1MDEgMTYgMTggMTYgQyAxNy4yNjc0OCAxNiAxNi42MDE1OTMgMTYuMjc5MzI4IDE2LjA3ODEyNSAxNi43MjI2NTYgTCA4Ljk0MzM1OTQgMTIuNTU4NTk0IEMgOC45NzgyMDk1IDEyLjM3NzEyMiA5IDEyLjE5MDk1MyA5IDEyIEMgOSAxMS44MDkwNDcgOC45NzgyMDk1IDExLjYyMjg3OCA4Ljk0MzM1OTQgMTEuNDQxNDA2IEwgMTYuMDc4MTI1IDcuMjc5Mjk2OSBDIDE2LjYwMTQ2IDcuNzIyMTQ2MSAxNy4yNjc5MjMgOCAxOCA4IEMgMTkuNjQ1MDEgOCAyMSA2LjY0NTAwOTYgMjEgNSBDIDIxIDMuMzU0OTkwNCAxOS42NDUwMSAyIDE4IDIgeiBNIDE4IDQgQyAxOC41NjQxMjkgNCAxOSA0LjQzNTg3MDYgMTkgNSBDIDE5IDUuNTY0MTI5NCAxOC41NjQxMjkgNiAxOCA2IEMgMTcuNDM1ODcxIDYgMTcgNS41NjQxMjk0IDE3IDUgQyAxNyA0LjQzNTg3MDYgMTcuNDM1ODcxIDQgMTggNCB6IE0gNiAxMSBDIDYuNTY0MTI5NCAxMSA3IDExLjQzNTg3MSA3IDEyIEMgNyAxMi41NjQxMjkgNi41NjQxMjk0IDEzIDYgMTMgQyA1LjQzNTg3MDYgMTMgNSAxMi41NjQxMjkgNSAxMiBDIDUgMTEuNDM1ODcxIDUuNDM1ODcwNiAxMSA2IDExIHogTSAxOCAxOCBDIDE4LjU2NDEyOSAxOCAxOSAxOC40MzU4NzEgMTkgMTkgQyAxOSAxOS41NjQxMjkgMTguNTY0MTI5IDIwIDE4IDIwIEMgMTcuNDM1ODcxIDIwIDE3IDE5LjU2NDEyOSAxNyAxOSBDIDE3IDE4LjQzNTg3MSAxNy40MzU4NzEgMTggMTggMTggeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-spreadsheet: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNENBRjUwIiBkPSJNMi4yIDIuMnYxNy42aDE3LjZWMi4ySDIuMnptMTUuNCA3LjdoLTUuNVY0LjRoNS41djUuNXpNOS45IDQuNHY1LjVINC40VjQuNGg1LjV6bS01LjUgNy43aDUuNXY1LjVINC40di01LjV6bTcuNyA1LjV2LTUuNWg1LjV2NS41aC01LjV6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-stop: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik02IDZoMTJ2MTJINnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tab: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIxIDNIM2MtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxOGMxLjEgMCAyLS45IDItMlY1YzAtMS4xLS45LTItMi0yem0wIDE2SDNWNWgxMHY0aDh2MTB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-table-rows: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMSw4SDNWNGgxOFY4eiBNMjEsMTBIM3Y0aDE4VjEweiBNMjEsMTZIM3Y0aDE4VjE2eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-tag: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCA0MyAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTI4LjgzMzIgMTIuMzM0TDMyLjk5OTggMTYuNTAwN0wzNy4xNjY1IDEyLjMzNEgyOC44MzMyWiIvPgoJCTxwYXRoIGQ9Ik0xNi4yMDk1IDIxLjYxMDRDMTUuNjg3MyAyMi4xMjk5IDE0Ljg0NDMgMjIuMTI5OSAxNC4zMjQ4IDIxLjYxMDRMNi45ODI5IDE0LjcyNDVDNi41NzI0IDE0LjMzOTQgNi4wODMxMyAxMy42MDk4IDYuMDQ3ODYgMTMuMDQ4MkM1Ljk1MzQ3IDExLjUyODggNi4wMjAwMiA4LjYxOTQ0IDYuMDY2MjEgNy4wNzY5NUM2LjA4MjgxIDYuNTE0NzcgNi41NTU0OCA2LjA0MzQ3IDcuMTE4MDQgNi4wMzA1NUM5LjA4ODYzIDUuOTg0NzMgMTMuMjYzOCA1LjkzNTc5IDEzLjY1MTggNi4zMjQyNUwyMS43MzY5IDEzLjYzOUMyMi4yNTYgMTQuMTU4NSAyMS43ODUxIDE1LjQ3MjQgMjEuMjYyIDE1Ljk5NDZMMTYuMjA5NSAyMS42MTA0Wk05Ljc3NTg1IDguMjY1QzkuMzM1NTEgNy44MjU2NiA4LjYyMzUxIDcuODI1NjYgOC4xODI4IDguMjY1QzcuNzQzNDYgOC43MDU3MSA3Ljc0MzQ2IDkuNDE3MzMgOC4xODI4IDkuODU2NjdDOC42MjM4MiAxMC4yOTY0IDkuMzM1ODIgMTAuMjk2NCA5Ljc3NTg1IDkuODU2NjdDMTAuMjE1NiA5LjQxNzMzIDEwLjIxNTYgOC43MDUzMyA5Ljc3NTg1IDguMjY1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-terminal: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiA+CiAgICA8cmVjdCBjbGFzcz0ianAtdGVybWluYWwtaWNvbi1iYWNrZ3JvdW5kLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZSIgd2lkdGg9IjIwIiBoZWlnaHQ9IjIwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyIDIpIiBmaWxsPSIjMzMzMzMzIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtdGVybWluYWwtaWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUtaW52ZXJzZSIgZD0iTTUuMDU2NjQgOC43NjE3MkM1LjA1NjY0IDguNTk3NjYgNS4wMzEyNSA4LjQ1MzEyIDQuOTgwNDcgOC4zMjgxMkM0LjkzMzU5IDguMTk5MjIgNC44NTU0NyA4LjA4MjAzIDQuNzQ2MDkgNy45NzY1NkM0LjY0MDYyIDcuODcxMDkgNC41IDcuNzc1MzkgNC4zMjQyMiA3LjY4OTQ1QzQuMTUyMzQgNy41OTk2MSAzLjk0MzM2IDcuNTExNzIgMy42OTcyNyA3LjQyNTc4QzMuMzAyNzMgNy4yODUxNiAyLjk0MzM2IDcuMTM2NzIgMi42MTkxNCA2Ljk4MDQ3QzIuMjk0OTIgNi44MjQyMiAyLjAxNzU4IDYuNjQyNTggMS43ODcxMSA2LjQzNTU1QzEuNTYwNTUgNi4yMjg1MiAxLjM4NDc3IDUuOTg4MjggMS4yNTk3NyA1LjcxNDg0QzEuMTM0NzcgNS40Mzc1IDEuMDcyMjcgNS4xMDkzOCAxLjA3MjI3IDQuNzMwNDdDMS4wNzIyNyA0LjM5ODQ0IDEuMTI4OTEgNC4wOTU3IDEuMjQyMTkgMy44MjIyN0MxLjM1NTQ3IDMuNTQ0OTIgMS41MTU2MiAzLjMwNDY5IDEuNzIyNjYgMy4xMDE1NkMxLjkyOTY5IDIuODk4NDQgMi4xNzk2OSAyLjczNDM3IDIuNDcyNjYgMi42MDkzOEMyLjc2NTYyIDIuNDg0MzggMy4wOTE4IDIuNDA0MyAzLjQ1MTE3IDIuMzY5MTRWMS4xMDkzOEg0LjM4ODY3VjIuMzgwODZDNC43NDAyMyAyLjQyNzczIDUuMDU2NjQgMi41MjM0NCA1LjMzNzg5IDIuNjY3OTdDNS42MTkxNCAyLjgxMjUgNS44NTc0MiAzLjAwMTk1IDYuMDUyNzMgMy4yMzYzM0M2LjI1MTk1IDMuNDY2OCA2LjQwNDMgMy43NDAyMyA2LjUwOTc3IDQuMDU2NjRDNi42MTkxNCA0LjM2OTE0IDYuNjczODMgNC43MjA3IDYuNjczODMgNS4xMTEzM0g1LjA0NDkyQzUuMDQ0OTIgNC42Mzg2NyA0LjkzNzUgNC4yODEyNSA0LjcyMjY2IDQuMDM5MDZDNC41MDc4MSAzLjc5Mjk3IDQuMjE2OCAzLjY2OTkyIDMuODQ5NjEgMy42Njk5MkMzLjY1MDM5IDMuNjY5OTIgMy40NzY1NiAzLjY5NzI3IDMuMzI4MTIgMy43NTE5NUMzLjE4MzU5IDMuODAyNzMgMy4wNjQ0NSAzLjg3Njk1IDIuOTcwNyAzLjk3NDYxQzIuODc2OTUgNC4wNjgzNiAyLjgwNjY0IDQuMTc5NjkgMi43NTk3NyA0LjMwODU5QzIuNzE2OCA0LjQzNzUgMi42OTUzMSA0LjU3ODEyIDIuNjk1MzEgNC43MzA0N0MyLjY5NTMxIDQuODgyODEgMi43MTY4IDUuMDE5NTMgMi43NTk3NyA1LjE0MDYyQzIuODA2NjQgNS4yNTc4MSAyLjg4MjgxIDUuMzY3MTkgMi45ODgyOCA1LjQ2ODc1QzMuMDk3NjYgNS41NzAzMSAzLjI0MDIzIDUuNjY3OTcgMy40MTYwMiA1Ljc2MTcyQzMuNTkxOCA1Ljg1MTU2IDMuODEwNTUgNS45NDMzNiA0LjA3MjI3IDYuMDM3MTFDNC40NjY4IDYuMTg1NTUgNC44MjQyMiA2LjMzOTg0IDUuMTQ0NTMgNi41QzUuNDY0ODQgNi42NTYyNSA1LjczODI4IDYuODM5ODQgNS45NjQ4NCA3LjA1MDc4QzYuMTk1MzEgNy4yNTc4MSA2LjM3MTA5IDcuNSA2LjQ5MjE5IDcuNzc3MzRDNi42MTcxOSA4LjA1MDc4IDYuNjc5NjkgOC4zNzUgNi42Nzk2OSA4Ljc1QzYuNjc5NjkgOS4wOTM3NSA2LjYyMzA1IDkuNDA0MyA2LjUwOTc3IDkuNjgxNjRDNi4zOTY0OCA5Ljk1NTA4IDYuMjM0MzggMTAuMTkxNCA2LjAyMzQ0IDEwLjM5MDZDNS44MTI1IDEwLjU4OTggNS41NTg1OSAxMC43NSA1LjI2MTcyIDEwLjg3MTFDNC45NjQ4NCAxMC45ODgzIDQuNjMyODEgMTEuMDY0NSA0LjI2NTYyIDExLjA5OTZWMTIuMjQ4SDMuMzMzOThWMTEuMDk5NkMzLjAwMTk1IDExLjA2ODQgMi42Nzk2OSAxMC45OTYxIDIuMzY3MTkgMTAuODgyOEMyLjA1NDY5IDEwLjc2NTYgMS43NzczNCAxMC41OTc3IDEuNTM1MTYgMTAuMzc4OUMxLjI5Njg4IDEwLjE2MDIgMS4xMDU0NyA5Ljg4NDc3IDAuOTYwOTM4IDkuNTUyNzNDMC44MTY0MDYgOS4yMTY4IDAuNzQ0MTQxIDguODE0NDUgMC43NDQxNDEgOC4zNDU3SDIuMzc4OTFDMi4zNzg5MSA4LjYyNjk1IDIuNDE5OTIgOC44NjMyOCAyLjUwMTk1IDkuMDU0NjlDMi41ODM5OCA5LjI0MjE5IDIuNjg5NDUgOS4zOTI1OCAyLjgxODM2IDkuNTA1ODZDMi45NTExNyA5LjYxNTIzIDMuMTAxNTYgOS42OTMzNiAzLjI2OTUzIDkuNzQwMjNDMy40Mzc1IDkuNzg3MTEgMy42MDkzOCA5LjgxMDU1IDMuNzg1MTYgOS44MTA1NUM0LjIwMzEyIDkuODEwNTUgNC41MTk1MyA5LjcxMjg5IDQuNzM0MzggOS41MTc1OEM0Ljk0OTIyIDkuMzIyMjcgNS4wNTY2NCA5LjA3MDMxIDUuMDU2NjQgOC43NjE3MlpNMTMuNDE4IDEyLjI3MTVIOC4wNzQyMlYxMUgxMy40MThWMTIuMjcxNVoiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMuOTUyNjQgNikiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPgo=);
  --jp-icon-text-editor: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtdGV4dC1lZGl0b3ItaWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xNSAxNUgzdjJoMTJ2LTJ6bTAtOEgzdjJoMTJWN3pNMyAxM2gxOHYtMkgzdjJ6bTAgOGgxOHYtMkgzdjJ6TTMgM3YyaDE4VjNIM3oiLz4KPC9zdmc+Cg==);
  --jp-icon-toc: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik03LDVIMjFWN0g3VjVNNywxM1YxMUgyMVYxM0g3TTQsNC41QTEuNSwxLjUgMCAwLDEgNS41LDZBMS41LDEuNSAwIDAsMSA0LDcuNUExLjUsMS41IDAgMCwxIDIuNSw2QTEuNSwxLjUgMCAwLDEgNCw0LjVNNCwxMC41QTEuNSwxLjUgMCAwLDEgNS41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMy41QTEuNSwxLjUgMCAwLDEgMi41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMC41TTcsMTlWMTdIMjFWMTlIN000LDE2LjVBMS41LDEuNSAwIDAsMSA1LjUsMThBMS41LDEuNSAwIDAsMSA0LDE5LjVBMS41LDEuNSAwIDAsMSAyLjUsMThBMS41LDEuNSAwIDAsMSA0LDE2LjVaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tree-view: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMiAxMVYzaC03djNIOVYzSDJ2OGg3VjhoMnYxMGg0djNoN3YtOGgtN3YzaC0yVjhoMnYzeiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMiAxNy4xODQ0IDIuOTY5NjggMTQuMzAzMiAxLjg2MDk0IDExLjQ0MDlaIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiMzMzMzMzMiIHN0cm9rZT0iIzMzMzMzMyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOCA5Ljg2NzE5KSIgZD0iTTIuODYwMTUgNC44NjUzNUwwLjcyNjU0OSAyLjk5OTU5TDAgMy42MzA0NUwyLjg2MDE1IDYuMTMxNTdMOCAwLjYzMDg3Mkw3LjI3ODU3IDBMMi44NjAxNSA0Ljg2NTM1WiIvPgo8L3N2Zz4K);
  --jp-icon-undo: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjUgOGMtMi42NSAwLTUuMDUuOTktNi45IDIuNkwyIDd2OWg5bC0zLjYyLTMuNjJjMS4zOS0xLjE2IDMuMTYtMS44OCA1LjEyLTEuODggMy41NCAwIDYuNTUgMi4zMSA3LjYgNS41bDIuMzctLjc4QzIxLjA4IDExLjAzIDE3LjE1IDggMTIuNSA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-user: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE2IDdhNCA0IDAgMTEtOCAwIDQgNCAwIDAxOCAwek0xMiAxNGE3IDcgMCAwMC03IDdoMTRhNyA3IDAgMDAtNy03eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-users: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZlcnNpb249IjEuMSIgdmlld0JveD0iMCAwIDM2IDI0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogPGcgY2xhc3M9ImpwLWljb24zIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjczMjcgMCAwIDEuNzMyNyAtMy42MjgyIC4wOTk1NzcpIiBmaWxsPSIjNjE2MTYxIj4KICA8cGF0aCB0cmFuc2Zvcm09Im1hdHJpeCgxLjUsMCwwLDEuNSwwLC02KSIgZD0ibTEyLjE4NiA3LjUwOThjLTEuMDUzNSAwLTEuOTc1NyAwLjU2NjUtMi40Nzg1IDEuNDEwMiAwLjc1MDYxIDAuMzEyNzcgMS4zOTc0IDAuODI2NDggMS44NzMgMS40NzI3aDMuNDg2M2MwLTEuNTkyLTEuMjg4OS0yLjg4MjgtMi44ODA5LTIuODgyOHoiLz4KICA8cGF0aCBkPSJtMjAuNDY1IDIuMzg5NWEyLjE4ODUgMi4xODg1IDAgMCAxLTIuMTg4NCAyLjE4ODUgMi4xODg1IDIuMTg4NSAwIDAgMS0yLjE4ODUtMi4xODg1IDIuMTg4NSAyLjE4ODUgMCAwIDEgMi4xODg1LTIuMTg4NSAyLjE4ODUgMi4xODg1IDAgMCAxIDIuMTg4NCAyLjE4ODV6Ii8+CiAgPHBhdGggdHJhbnNmb3JtPSJtYXRyaXgoMS41LDAsMCwxLjUsMCwtNikiIGQ9Im0zLjU4OTggOC40MjE5Yy0xLjExMjYgMC0yLjAxMzcgMC45MDExMS0yLjAxMzcgMi4wMTM3aDIuODE0NWMwLjI2Nzk3LTAuMzczMDkgMC41OTA3LTAuNzA0MzUgMC45NTg5OC0wLjk3ODUyLTAuMzQ0MzMtMC42MTY4OC0xLjAwMzEtMS4wMzUyLTEuNzU5OC0xLjAzNTJ6Ii8+CiAgPHBhdGggZD0ibTYuOTE1NCA0LjYyM2ExLjUyOTQgMS41Mjk0IDAgMCAxLTEuNTI5NCAxLjUyOTQgMS41Mjk0IDEuNTI5NCAwIDAgMS0xLjUyOTQtMS41Mjk0IDEuNTI5NCAxLjUyOTQgMCAwIDEgMS41Mjk0LTEuNTI5NCAxLjUyOTQgMS41Mjk0IDAgMCAxIDEuNTI5NCAxLjUyOTR6Ii8+CiAgPHBhdGggZD0ibTYuMTM1IDEzLjUzNWMwLTMuMjM5MiAyLjYyNTktNS44NjUgNS44NjUtNS44NjUgMy4yMzkyIDAgNS44NjUgMi42MjU5IDUuODY1IDUuODY1eiIvPgogIDxjaXJjbGUgY3g9IjEyIiBjeT0iMy43Njg1IiByPSIyLjk2ODUiLz4KIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-vega: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbjEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjEyMTIxIj4KICAgIDxwYXRoIGQ9Ik0xMC42IDUuNGwyLjItMy4ySDIuMnY3LjNsNC02LjZ6Ii8+CiAgICA8cGF0aCBkPSJNMTUuOCAyLjJsLTQuNCA2LjZMNyA2LjNsLTQuOCA4djUuNWgxNy42VjIuMmgtNHptLTcgMTUuNEg1LjV2LTQuNGgzLjN2NC40em00LjQgMEg5LjhWOS44aDMuNHY3Ljh6bTQuNCAwaC0zLjRWNi41aDMuNHYxMS4xeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-word: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KIDxnIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzQxNDE0MSI+CiAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiA8L2c+CiA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSguNDMgLjA0MDEpIiBmaWxsPSIjZmZmIj4KICA8cGF0aCBkPSJtNC4xNCA4Ljc2cTAuMDY4Mi0xLjg5IDIuNDItMS44OSAxLjE2IDAgMS42OCAwLjQyIDAuNTY3IDAuNDEgMC41NjcgMS4xNnYzLjQ3cTAgMC40NjIgMC41MTQgMC40NjIgMC4xMDMgMCAwLjItMC4wMjMxdjAuNzE0cS0wLjM5OSAwLjEwMy0wLjY1MSAwLjEwMy0wLjQ1MiAwLTAuNjkzLTAuMjItMC4yMzEtMC4yLTAuMjg0LTAuNjYyLTAuOTU2IDAuODcyLTIgMC44NzItMC45MDMgMC0xLjQ3LTAuNDcyLTAuNTI1LTAuNDcyLTAuNTI1LTEuMjYgMC0wLjI2MiAwLjA0NTItMC40NzIgMC4wNTY3LTAuMjIgMC4xMTYtMC4zNzggMC4wNjgyLTAuMTY4IDAuMjMxLTAuMzA0IDAuMTU4LTAuMTQ3IDAuMjYyLTAuMjQyIDAuMTE2LTAuMDkxNCAwLjM2OC0wLjE2OCAwLjI2Mi0wLjA5MTQgMC4zOTktMC4xMjYgMC4xMzYtMC4wNDUyIDAuNDcyLTAuMTAzIDAuMzM2LTAuMDU3OCAwLjUwNC0wLjA3OTggMC4xNTgtMC4wMjMxIDAuNTY3LTAuMDc5OCAwLjU1Ni0wLjA2ODIgMC43NzctMC4yMjEgMC4yMi0wLjE1MiAwLjIyLTAuNDQxdi0wLjI1MnEwLTAuNDMtMC4zNTctMC42NjItMC4zMzYtMC4yMzEtMC45NzYtMC4yMzEtMC42NjIgMC0wLjk5OCAwLjI2Mi0wLjMzNiAwLjI1Mi0wLjM5OSAwLjc5OHptMS44OSAzLjY4cTAuNzg4IDAgMS4yNi0wLjQxIDAuNTA0LTAuNDIgMC41MDQtMC45MDN2LTEuMDVxLTAuMjg0IDAuMTM2LTAuODYxIDAuMjMxLTAuNTY3IDAuMDkxNC0wLjk4NyAwLjE1OC0wLjQyIDAuMDY4Mi0wLjc2NiAwLjMyNi0wLjMzNiAwLjI1Mi0wLjMzNiAwLjcwNHQwLjMwNCAwLjcwNCAwLjg2MSAwLjI1MnoiIHN0cm9rZS13aWR0aD0iMS4wNSIvPgogIDxwYXRoIGQ9Im0xMCA0LjU2aDAuOTQ1djMuMTVxMC42NTEtMC45NzYgMS44OS0wLjk3NiAxLjE2IDAgMS44OSAwLjg0IDAuNjgyIDAuODQgMC42ODIgMi4zMSAwIDEuNDctMC43MDQgMi40Mi0wLjcwNCAwLjg4Mi0xLjg5IDAuODgyLTEuMjYgMC0xLjg5LTEuMDJ2MC43NjZoLTAuODV6bTIuNjIgMy4wNHEtMC43NDYgMC0xLjE2IDAuNjQtMC40NTIgMC42My0wLjQ1MiAxLjY4IDAgMS4wNSAwLjQ1MiAxLjY4dDEuMTYgMC42M3EwLjc3NyAwIDEuMjYtMC42MyAwLjQ5NC0wLjY0IDAuNDk0LTEuNjggMC0xLjA1LTAuNDcyLTEuNjgtMC40NjItMC42NC0xLjI2LTAuNjR6IiBzdHJva2Utd2lkdGg9IjEuMDUiLz4KICA8cGF0aCBkPSJtMi43MyAxNS44IDEzLjYgMC4wMDgxYzAuMDA2OSAwIDAtMi42IDAtMi42IDAtMC4wMDc4LTEuMTUgMC0xLjE1IDAtMC4wMDY5IDAtMC4wMDgzIDEuNS0wLjAwODMgMS41LTJlLTMgLTAuMDAxNC0xMS4zLTAuMDAxNC0xMS4zLTAuMDAxNGwtMC4wMDU5Mi0xLjVjMC0wLjAwNzgtMS4xNyAwLjAwMTMtMS4xNyAwLjAwMTN6IiBzdHJva2Utd2lkdGg9Ii45NzUiLz4KIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-yaml: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1jb250cmFzdDIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjRDgxQjYwIj4KICAgIDxwYXRoIGQ9Ik03LjIgMTguNnYtNS40TDMgNS42aDMuM2wxLjQgMy4xYy4zLjkuNiAxLjYgMSAyLjUuMy0uOC42LTEuNiAxLTIuNWwxLjQtMy4xaDMuNGwtNC40IDcuNnY1LjVsLTIuOS0uMXoiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxNi41IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxMSIgcj0iMi4xIi8+CiAgPC9nPgo8L3N2Zz4K);
}

/* Icon CSS class declarations */

.jp-AddAboveIcon {
  background-image: var(--jp-icon-add-above);
}

.jp-AddBelowIcon {
  background-image: var(--jp-icon-add-below);
}

.jp-AddIcon {
  background-image: var(--jp-icon-add);
}

.jp-BellIcon {
  background-image: var(--jp-icon-bell);
}

.jp-BugDotIcon {
  background-image: var(--jp-icon-bug-dot);
}

.jp-BugIcon {
  background-image: var(--jp-icon-bug);
}

.jp-BuildIcon {
  background-image: var(--jp-icon-build);
}

.jp-CaretDownEmptyIcon {
  background-image: var(--jp-icon-caret-down-empty);
}

.jp-CaretDownEmptyThinIcon {
  background-image: var(--jp-icon-caret-down-empty-thin);
}

.jp-CaretDownIcon {
  background-image: var(--jp-icon-caret-down);
}

.jp-CaretLeftIcon {
  background-image: var(--jp-icon-caret-left);
}

.jp-CaretRightIcon {
  background-image: var(--jp-icon-caret-right);
}

.jp-CaretUpEmptyThinIcon {
  background-image: var(--jp-icon-caret-up-empty-thin);
}

.jp-CaretUpIcon {
  background-image: var(--jp-icon-caret-up);
}

.jp-CaseSensitiveIcon {
  background-image: var(--jp-icon-case-sensitive);
}

.jp-CheckIcon {
  background-image: var(--jp-icon-check);
}

.jp-CircleEmptyIcon {
  background-image: var(--jp-icon-circle-empty);
}

.jp-CircleIcon {
  background-image: var(--jp-icon-circle);
}

.jp-ClearIcon {
  background-image: var(--jp-icon-clear);
}

.jp-CloseIcon {
  background-image: var(--jp-icon-close);
}

.jp-CodeCheckIcon {
  background-image: var(--jp-icon-code-check);
}

.jp-CodeIcon {
  background-image: var(--jp-icon-code);
}

.jp-CollapseAllIcon {
  background-image: var(--jp-icon-collapse-all);
}

.jp-ConsoleIcon {
  background-image: var(--jp-icon-console);
}

.jp-CopyIcon {
  background-image: var(--jp-icon-copy);
}

.jp-CopyrightIcon {
  background-image: var(--jp-icon-copyright);
}

.jp-CutIcon {
  background-image: var(--jp-icon-cut);
}

.jp-DeleteIcon {
  background-image: var(--jp-icon-delete);
}

.jp-DownloadIcon {
  background-image: var(--jp-icon-download);
}

.jp-DuplicateIcon {
  background-image: var(--jp-icon-duplicate);
}

.jp-EditIcon {
  background-image: var(--jp-icon-edit);
}

.jp-EllipsesIcon {
  background-image: var(--jp-icon-ellipses);
}

.jp-ErrorIcon {
  background-image: var(--jp-icon-error);
}

.jp-ExpandAllIcon {
  background-image: var(--jp-icon-expand-all);
}

.jp-ExtensionIcon {
  background-image: var(--jp-icon-extension);
}

.jp-FastForwardIcon {
  background-image: var(--jp-icon-fast-forward);
}

.jp-FileIcon {
  background-image: var(--jp-icon-file);
}

.jp-FileUploadIcon {
  background-image: var(--jp-icon-file-upload);
}

.jp-FilterDotIcon {
  background-image: var(--jp-icon-filter-dot);
}

.jp-FilterIcon {
  background-image: var(--jp-icon-filter);
}

.jp-FilterListIcon {
  background-image: var(--jp-icon-filter-list);
}

.jp-FolderFavoriteIcon {
  background-image: var(--jp-icon-folder-favorite);
}

.jp-FolderIcon {
  background-image: var(--jp-icon-folder);
}

.jp-HomeIcon {
  background-image: var(--jp-icon-home);
}

.jp-Html5Icon {
  background-image: var(--jp-icon-html5);
}

.jp-ImageIcon {
  background-image: var(--jp-icon-image);
}

.jp-InfoIcon {
  background-image: var(--jp-icon-info);
}

.jp-InspectorIcon {
  background-image: var(--jp-icon-inspector);
}

.jp-JsonIcon {
  background-image: var(--jp-icon-json);
}

.jp-JuliaIcon {
  background-image: var(--jp-icon-julia);
}

.jp-JupyterFaviconIcon {
  background-image: var(--jp-icon-jupyter-favicon);
}

.jp-JupyterIcon {
  background-image: var(--jp-icon-jupyter);
}

.jp-JupyterlabWordmarkIcon {
  background-image: var(--jp-icon-jupyterlab-wordmark);
}

.jp-KernelIcon {
  background-image: var(--jp-icon-kernel);
}

.jp-KeyboardIcon {
  background-image: var(--jp-icon-keyboard);
}

.jp-LaunchIcon {
  background-image: var(--jp-icon-launch);
}

.jp-LauncherIcon {
  background-image: var(--jp-icon-launcher);
}

.jp-LineFormIcon {
  background-image: var(--jp-icon-line-form);
}

.jp-LinkIcon {
  background-image: var(--jp-icon-link);
}

.jp-ListIcon {
  background-image: var(--jp-icon-list);
}

.jp-MarkdownIcon {
  background-image: var(--jp-icon-markdown);
}

.jp-MoveDownIcon {
  background-image: var(--jp-icon-move-down);
}

.jp-MoveUpIcon {
  background-image: var(--jp-icon-move-up);
}

.jp-NewFolderIcon {
  background-image: var(--jp-icon-new-folder);
}

.jp-NotTrustedIcon {
  background-image: var(--jp-icon-not-trusted);
}

.jp-NotebookIcon {
  background-image: var(--jp-icon-notebook);
}

.jp-NumberingIcon {
  background-image: var(--jp-icon-numbering);
}

.jp-OfflineBoltIcon {
  background-image: var(--jp-icon-offline-bolt);
}

.jp-PaletteIcon {
  background-image: var(--jp-icon-palette);
}

.jp-PasteIcon {
  background-image: var(--jp-icon-paste);
}

.jp-PdfIcon {
  background-image: var(--jp-icon-pdf);
}

.jp-PythonIcon {
  background-image: var(--jp-icon-python);
}

.jp-RKernelIcon {
  background-image: var(--jp-icon-r-kernel);
}

.jp-ReactIcon {
  background-image: var(--jp-icon-react);
}

.jp-RedoIcon {
  background-image: var(--jp-icon-redo);
}

.jp-RefreshIcon {
  background-image: var(--jp-icon-refresh);
}

.jp-RegexIcon {
  background-image: var(--jp-icon-regex);
}

.jp-RunIcon {
  background-image: var(--jp-icon-run);
}

.jp-RunningIcon {
  background-image: var(--jp-icon-running);
}

.jp-SaveIcon {
  background-image: var(--jp-icon-save);
}

.jp-SearchIcon {
  background-image: var(--jp-icon-search);
}

.jp-SettingsIcon {
  background-image: var(--jp-icon-settings);
}

.jp-ShareIcon {
  background-image: var(--jp-icon-share);
}

.jp-SpreadsheetIcon {
  background-image: var(--jp-icon-spreadsheet);
}

.jp-StopIcon {
  background-image: var(--jp-icon-stop);
}

.jp-TabIcon {
  background-image: var(--jp-icon-tab);
}

.jp-TableRowsIcon {
  background-image: var(--jp-icon-table-rows);
}

.jp-TagIcon {
  background-image: var(--jp-icon-tag);
}

.jp-TerminalIcon {
  background-image: var(--jp-icon-terminal);
}

.jp-TextEditorIcon {
  background-image: var(--jp-icon-text-editor);
}

.jp-TocIcon {
  background-image: var(--jp-icon-toc);
}

.jp-TreeViewIcon {
  background-image: var(--jp-icon-tree-view);
}

.jp-TrustedIcon {
  background-image: var(--jp-icon-trusted);
}

.jp-UndoIcon {
  background-image: var(--jp-icon-undo);
}

.jp-UserIcon {
  background-image: var(--jp-icon-user);
}

.jp-UsersIcon {
  background-image: var(--jp-icon-users);
}

.jp-VegaIcon {
  background-image: var(--jp-icon-vega);
}

.jp-WordIcon {
  background-image: var(--jp-icon-word);
}

.jp-YamlIcon {
  background-image: var(--jp-icon-yaml);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

.jp-Icon,
.jp-MaterialIcon {
  background-position: center;
  background-repeat: no-repeat;
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-cover {
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

/**
 * (DEPRECATED) Support for specific CSS icon sizes
 */

.jp-Icon-16 {
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-18 {
  background-size: 18px;
  min-width: 18px;
  min-height: 18px;
}

.jp-Icon-20 {
  background-size: 20px;
  min-width: 20px;
  min-height: 20px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.lm-TabBar .lm-TabBar-addButton {
  align-items: center;
  display: flex;
  padding: 4px;
  padding-bottom: 5px;
  margin-right: 1px;
  background-color: var(--jp-layout-color2);
}

.lm-TabBar .lm-TabBar-addButton:hover {
  background-color: var(--jp-layout-color1);
}

.lm-DockPanel-tabBar .lm-TabBar-tab {
  width: var(--jp-private-horizontal-tab-width);
}

.lm-DockPanel-tabBar .lm-TabBar-content {
  flex: unset;
}

.lm-DockPanel-tabBar[data-orientation='horizontal'] {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for icons as inline SVG HTMLElements
 */

/* recolor the primary elements of an icon */
.jp-icon0[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon1[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon2[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-accent0[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-accent1[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-accent2[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-accent3[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-accent4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-accent0[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-accent1[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-accent2[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-accent3[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-accent4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-none[fill] {
  fill: none;
}

.jp-icon-none[stroke] {
  stroke: none;
}

/* brand icon colors. Same for light and dark */
.jp-icon-brand0[fill] {
  fill: var(--jp-brand-color0);
}

.jp-icon-brand1[fill] {
  fill: var(--jp-brand-color1);
}

.jp-icon-brand2[fill] {
  fill: var(--jp-brand-color2);
}

.jp-icon-brand3[fill] {
  fill: var(--jp-brand-color3);
}

.jp-icon-brand4[fill] {
  fill: var(--jp-brand-color4);
}

.jp-icon-brand0[stroke] {
  stroke: var(--jp-brand-color0);
}

.jp-icon-brand1[stroke] {
  stroke: var(--jp-brand-color1);
}

.jp-icon-brand2[stroke] {
  stroke: var(--jp-brand-color2);
}

.jp-icon-brand3[stroke] {
  stroke: var(--jp-brand-color3);
}

.jp-icon-brand4[stroke] {
  stroke: var(--jp-brand-color4);
}

/* warn icon colors. Same for light and dark */
.jp-icon-warn0[fill] {
  fill: var(--jp-warn-color0);
}

.jp-icon-warn1[fill] {
  fill: var(--jp-warn-color1);
}

.jp-icon-warn2[fill] {
  fill: var(--jp-warn-color2);
}

.jp-icon-warn3[fill] {
  fill: var(--jp-warn-color3);
}

.jp-icon-warn0[stroke] {
  stroke: var(--jp-warn-color0);
}

.jp-icon-warn1[stroke] {
  stroke: var(--jp-warn-color1);
}

.jp-icon-warn2[stroke] {
  stroke: var(--jp-warn-color2);
}

.jp-icon-warn3[stroke] {
  stroke: var(--jp-warn-color3);
}

/* icon colors that contrast well with each other and most backgrounds */
.jp-icon-contrast0[fill] {
  fill: var(--jp-icon-contrast-color0);
}

.jp-icon-contrast1[fill] {
  fill: var(--jp-icon-contrast-color1);
}

.jp-icon-contrast2[fill] {
  fill: var(--jp-icon-contrast-color2);
}

.jp-icon-contrast3[fill] {
  fill: var(--jp-icon-contrast-color3);
}

.jp-icon-contrast0[stroke] {
  stroke: var(--jp-icon-contrast-color0);
}

.jp-icon-contrast1[stroke] {
  stroke: var(--jp-icon-contrast-color1);
}

.jp-icon-contrast2[stroke] {
  stroke: var(--jp-icon-contrast-color2);
}

.jp-icon-contrast3[stroke] {
  stroke: var(--jp-icon-contrast-color3);
}

.jp-icon-dot[fill] {
  fill: var(--jp-warn-color0);
}

.jp-jupyter-icon-color[fill] {
  fill: var(--jp-jupyter-icon-color, var(--jp-warn-color0));
}

.jp-notebook-icon-color[fill] {
  fill: var(--jp-notebook-icon-color, var(--jp-warn-color0));
}

.jp-json-icon-color[fill] {
  fill: var(--jp-json-icon-color, var(--jp-warn-color1));
}

.jp-console-icon-color[fill] {
  fill: var(--jp-console-icon-color, white);
}

.jp-console-icon-background-color[fill] {
  fill: var(--jp-console-icon-background-color, var(--jp-brand-color1));
}

.jp-terminal-icon-color[fill] {
  fill: var(--jp-terminal-icon-color, var(--jp-layout-color2));
}

.jp-terminal-icon-background-color[fill] {
  fill: var(
    --jp-terminal-icon-background-color,
    var(--jp-inverse-layout-color2)
  );
}

.jp-text-editor-icon-color[fill] {
  fill: var(--jp-text-editor-icon-color, var(--jp-inverse-layout-color3));
}

.jp-inspector-icon-color[fill] {
  fill: var(--jp-inspector-icon-color, var(--jp-inverse-layout-color3));
}

/* CSS for icons in selected filebrowser listing items */
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

.jp-DirListing-item.jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* stylelint-disable selector-max-class, selector-max-compound-selectors */

/**
* TODO: come up with non css-hack solution for showing the busy icon on top
*  of the close icon
* CSS for complex behavior of close icon of tabs in the main area tabbar
*/
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}

.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

/* stylelint-enable selector-max-class, selector-max-compound-selectors */

/* CSS for icons in status bar */
#jp-main-statusbar .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

#jp-main-statusbar .jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* special handling for splash icon CSS. While the theme CSS reloads during
   splash, the splash icon can loose theming. To prevent that, we set a
   default for its color variable */
:root {
  --jp-warn-color0: var(--md-orange-700);
}

/* not sure what to do with this one, used in filebrowser listing */
.jp-DragIcon {
  margin-right: 4px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for alt colors for icons as inline SVG HTMLElements
 */

/* alt recolor the primary elements of an icon */
.jp-icon-alt .jp-icon0[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-alt .jp-icon1[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-alt .jp-icon2[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-alt .jp-icon3[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-alt .jp-icon4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-alt .jp-icon0[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-alt .jp-icon1[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-alt .jp-icon2[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-alt .jp-icon3[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-alt .jp-icon4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* alt recolor the accent elements of an icon */
.jp-icon-alt .jp-icon-accent0[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-alt .jp-icon-accent1[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-alt .jp-icon-accent2[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-alt .jp-icon-accent3[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-alt .jp-icon-accent4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-alt .jp-icon-accent0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-alt .jp-icon-accent1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-alt .jp-icon-accent2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-alt .jp-icon-accent3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-alt .jp-icon-accent4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-icon-hoverShow:not(:hover) .jp-icon-hoverShow-content {
  display: none !important;
}

/**
 * Support for hover colors for icons as inline SVG HTMLElements
 */

/**
 * regular colors
 */

/* recolor the primary elements of an icon */
.jp-icon-hover :hover .jp-icon0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-hover :hover .jp-icon1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-hover :hover .jp-icon2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-hover :hover .jp-icon3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-hover :hover .jp-icon4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-hover :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-hover :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-hover :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-hover :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-hover :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-hover :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-hover :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-hover :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-hover :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-hover :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-hover :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-hover :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-hover :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-hover :hover .jp-icon-none-hover[fill] {
  fill: none;
}

.jp-icon-hover :hover .jp-icon-none-hover[stroke] {
  stroke: none;
}

/**
 * inverse colors
 */

/* inverse recolor the primary elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* inverse recolor the accent elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-IFrame {
  width: 100%;
  height: 100%;
}

.jp-IFrame > iframe {
  border: none;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-IFrame {
  position: relative;
}

body.lm-mod-override-cursor .jp-IFrame::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-HoverBox {
  position: fixed;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FormGroup-content fieldset {
  border: none;
  padding: 0;
  min-width: 0;
  width: 100%;
}

/* stylelint-disable selector-max-type */

.jp-FormGroup-content fieldset .jp-inputFieldWrapper input,
.jp-FormGroup-content fieldset .jp-inputFieldWrapper select,
.jp-FormGroup-content fieldset .jp-inputFieldWrapper textarea {
  font-size: var(--jp-content-font-size2);
  border-color: var(--jp-input-border-color);
  border-style: solid;
  border-radius: var(--jp-border-radius);
  border-width: 1px;
  padding: 6px 8px;
  background: none;
  color: var(--jp-ui-font-color0);
  height: inherit;
}

.jp-FormGroup-content fieldset input[type='checkbox'] {
  position: relative;
  top: 2px;
  margin-left: 0;
}

.jp-FormGroup-content button.jp-mod-styled {
  cursor: pointer;
}

.jp-FormGroup-content .checkbox label {
  cursor: pointer;
  font-size: var(--jp-content-font-size1);
}

.jp-FormGroup-content .jp-root > fieldset > legend {
  display: none;
}

.jp-FormGroup-content .jp-root > fieldset > p {
  display: none;
}

/** copy of `input.jp-mod-styled:focus` style */
.jp-FormGroup-content fieldset input:focus,
.jp-FormGroup-content fieldset select:focus {
  -moz-outline-radius: unset;
  outline: var(--jp-border-width) solid var(--md-blue-500);
  outline-offset: -1px;
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-FormGroup-content fieldset input:hover:not(:focus),
.jp-FormGroup-content fieldset select:hover:not(:focus) {
  background-color: var(--jp-border-color2);
}

/* stylelint-enable selector-max-type */

.jp-FormGroup-content .checkbox .field-description {
  /* Disable default description field for checkbox:
   because other widgets do not have description fields,
   we add descriptions to each widget on the field level.
  */
  display: none;
}

.jp-FormGroup-content #root__description {
  display: none;
}

.jp-FormGroup-content .jp-modifiedIndicator {
  width: 5px;
  background-color: var(--jp-brand-color2);
  margin-top: 0;
  margin-left: calc(var(--jp-private-settingeditor-modifier-indent) * -1);
  flex-shrink: 0;
}

.jp-FormGroup-content .jp-modifiedIndicator.jp-errorIndicator {
  background-color: var(--jp-error-color0);
  margin-right: 0.5em;
}

/* RJSF ARRAY style */

.jp-arrayFieldWrapper legend {
  font-size: var(--jp-content-font-size2);
  color: var(--jp-ui-font-color0);
  flex-basis: 100%;
  padding: 4px 0;
  font-weight: var(--jp-content-heading-font-weight);
  border-bottom: 1px solid var(--jp-border-color2);
}

.jp-arrayFieldWrapper .field-description {
  padding: 4px 0;
  white-space: pre-wrap;
}

.jp-arrayFieldWrapper .array-item {
  width: 100%;
  border: 1px solid var(--jp-border-color2);
  border-radius: 4px;
  margin: 4px;
}

.jp-ArrayOperations {
  display: flex;
  margin-left: 8px;
}

.jp-ArrayOperationsButton {
  margin: 2px;
}

.jp-ArrayOperationsButton .jp-icon3[fill] {
  fill: var(--jp-ui-font-color0);
}

button.jp-ArrayOperationsButton.jp-mod-styled:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

/* RJSF form validation error */

.jp-FormGroup-content .validationErrors {
  color: var(--jp-error-color0);
}

/* Hide panel level error as duplicated the field level error */
.jp-FormGroup-content .panel.errors {
  display: none;
}

/* RJSF normal content (settings-editor) */

.jp-FormGroup-contentNormal {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.jp-FormGroup-contentNormal .jp-FormGroup-contentItem {
  margin-left: 7px;
  color: var(--jp-ui-font-color0);
}

.jp-FormGroup-contentNormal .jp-FormGroup-description {
  flex-basis: 100%;
  padding: 4px 7px;
}

.jp-FormGroup-contentNormal .jp-FormGroup-default {
  flex-basis: 100%;
  padding: 4px 7px;
}

.jp-FormGroup-contentNormal .jp-FormGroup-fieldLabel {
  font-size: var(--jp-content-font-size1);
  font-weight: normal;
  min-width: 120px;
}

.jp-FormGroup-contentNormal fieldset:not(:first-child) {
  margin-left: 7px;
}

.jp-FormGroup-contentNormal .field-array-of-string .array-item {
  /* Display `jp-ArrayOperations` buttons side-by-side with content except
    for small screens where flex-wrap will place them one below the other.
  */
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.jp-FormGroup-contentNormal .jp-objectFieldWrapper .form-group {
  padding: 2px 8px 2px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 2px;
}

/* RJSF compact content (metadata-form) */

.jp-FormGroup-content.jp-FormGroup-contentCompact {
  width: 100%;
}

.jp-FormGroup-contentCompact .form-group {
  display: flex;
  padding: 0.5em 0.2em 0.5em 0;
}

.jp-FormGroup-contentCompact
  .jp-FormGroup-compactTitle
  .jp-FormGroup-description {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color2);
}

.jp-FormGroup-contentCompact .jp-FormGroup-fieldLabel {
  padding-bottom: 0.3em;
}

.jp-FormGroup-contentCompact .jp-inputFieldWrapper .form-control {
  width: 100%;
  box-sizing: border-box;
}

.jp-FormGroup-contentCompact .jp-arrayFieldWrapper .jp-FormGroup-compactTitle {
  padding-bottom: 7px;
}

.jp-FormGroup-contentCompact
  .jp-objectFieldWrapper
  .jp-objectFieldWrapper
  .form-group {
  padding: 2px 8px 2px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 2px;
}

.jp-FormGroup-contentCompact ul.error-detail {
  margin-block-start: 0.5em;
  margin-block-end: 0.5em;
  padding-inline-start: 1em;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-SidePanel {
  display: flex;
  flex-direction: column;
  min-width: var(--jp-sidebar-min-width);
  overflow-y: auto;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size1);
}

.jp-SidePanel-header {
  flex: 0 0 auto;
  display: flex;
  border-bottom: var(--jp-border-width) solid var(--jp-border-color2);
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin: 0;
  padding: 2px;
  text-transform: uppercase;
}

.jp-SidePanel-toolbar {
  flex: 0 0 auto;
}

.jp-SidePanel-content {
  flex: 1 1 auto;
}

.jp-SidePanel-toolbar,
.jp-AccordionPanel-toolbar {
  height: var(--jp-private-toolbar-height);
}

.jp-SidePanel-toolbar.jp-Toolbar-micro {
  display: none;
}

.lm-AccordionPanel .jp-AccordionPanel-title {
  box-sizing: border-box;
  line-height: 25px;
  margin: 0;
  display: flex;
  align-items: center;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  font-size: var(--jp-ui-font-size0);
}

.jp-AccordionPanel-title {
  cursor: pointer;
  user-select: none;
  -moz-user-select: none;
  -webkit-user-select: none;
  text-transform: uppercase;
}

.lm-AccordionPanel[data-orientation='horizontal'] > .jp-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}

.jp-AccordionPanel-title .lm-AccordionPanel-titleLabel {
  user-select: none;
  text-overflow: ellipsis;
  white-space: nowrap;
  overflow: hidden;
}

.jp-AccordionPanel-title .lm-AccordionPanel-titleCollapser {
  transform: rotate(-90deg);
  margin: auto 0;
  height: 16px;
}

.jp-AccordionPanel-title.lm-mod-expanded .lm-AccordionPanel-titleCollapser {
  transform: rotate(0deg);
}

.lm-AccordionPanel .jp-AccordionPanel-toolbar {
  background: none;
  box-shadow: none;
  border: none;
  margin-left: auto;
}

.lm-AccordionPanel .lm-SplitPanel-handle:hover {
  background: var(--jp-layout-color3);
}

.jp-text-truncated {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Spinner {
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-layout-color0);
  outline: none;
}

.jp-SpinnerContent {
  font-size: 10px;
  margin: 50px auto;
  text-indent: -9999em;
  width: 3em;
  height: 3em;
  border-radius: 50%;
  background: var(--jp-brand-color3);
  background: linear-gradient(
    to right,
    #f37626 10%,
    rgba(255, 255, 255, 0) 42%
  );
  position: relative;
  animation: load3 1s infinite linear, fadeIn 1s;
}

.jp-SpinnerContent::before {
  width: 50%;
  height: 50%;
  background: #f37626;
  border-radius: 100% 0 0;
  position: absolute;
  top: 0;
  left: 0;
  content: '';
}

.jp-SpinnerContent::after {
  background: var(--jp-layout-color0);
  width: 75%;
  height: 75%;
  border-radius: 50%;
  content: '';
  margin: auto;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

@keyframes fadeIn {
  0% {
    opacity: 0;
  }

  100% {
    opacity: 1;
  }
}

@keyframes load3 {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: none;
  box-sizing: border-box;
  text-align: center;
  line-height: 32px;
  height: 32px;
  padding: 0 12px;
  letter-spacing: 0.8px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled {
  background: var(--jp-input-background);
  height: 28px;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding-left: 7px;
  padding-right: 7px;
  font-size: var(--jp-ui-font-size2);
  color: var(--jp-ui-font-color0);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input[type='checkbox'].jp-mod-styled {
  appearance: checkbox;
  -webkit-appearance: checkbox;
  -moz-appearance: checkbox;
  height: auto;
}

input.jp-mod-styled:focus {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-select-wrapper {
  display: flex;
  position: relative;
  flex-direction: column;
  padding: 1px;
  background-color: var(--jp-layout-color1);
  box-sizing: border-box;
  margin-bottom: 12px;
}

.jp-select-wrapper:not(.multiple) {
  height: 28px;
}

.jp-select-wrapper.jp-mod-focused select.jp-mod-styled {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-input-active-background);
}

select.jp-mod-styled:hover {
  cursor: pointer;
  color: var(--jp-ui-font-color0);
  background-color: var(--jp-input-hover-background);
  box-shadow: inset 0 0 1px rgba(0, 0, 0, 0.5);
}

select.jp-mod-styled {
  flex: 1 1 auto;
  width: 100%;
  font-size: var(--jp-ui-font-size2);
  background: var(--jp-input-background);
  color: var(--jp-ui-font-color0);
  padding: 0 25px 0 8px;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

select.jp-mod-styled:not([multiple]) {
  height: 32px;
}

select.jp-mod-styled[multiple] {
  max-height: 200px;
  overflow-y: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-switch {
  display: flex;
  align-items: center;
  padding-left: 4px;
  padding-right: 4px;
  font-size: var(--jp-ui-font-size1);
  background-color: transparent;
  color: var(--jp-ui-font-color1);
  border: none;
  height: 20px;
}

.jp-switch:hover {
  background-color: var(--jp-layout-color2);
}

.jp-switch-label {
  margin-right: 5px;
  font-family: var(--jp-ui-font-family);
}

.jp-switch-track {
  cursor: pointer;
  background-color: var(--jp-switch-color, var(--jp-border-color1));
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 34px;
  height: 16px;
  width: 35px;
  position: relative;
}

.jp-switch-track::before {
  content: '';
  position: absolute;
  height: 10px;
  width: 10px;
  margin: 3px;
  left: 0;
  background-color: var(--jp-ui-inverse-font-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 50%;
}

.jp-switch[aria-checked='true'] .jp-switch-track {
  background-color: var(--jp-switch-true-position-color, var(--jp-warn-color0));
}

.jp-switch[aria-checked='true'] .jp-switch-track::before {
  /* track width (35) - margins (3 + 3) - thumb width (10) */
  left: 19px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toolbar-height: calc(
    28px + var(--jp-border-width)
  ); /* leave 28px for content */
}

.jp-Toolbar {
  color: var(--jp-ui-font-color1);
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: 2px;
  z-index: 8;
  overflow-x: hidden;
}

/* Toolbar items */

.jp-Toolbar > .jp-Toolbar-item.jp-Toolbar-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-Toolbar-item.jp-Toolbar-kernelStatus {
  display: inline-block;
  width: 32px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: 16px;
}

.jp-Toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  display: flex;
  padding-left: 1px;
  padding-right: 1px;
  font-size: var(--jp-ui-font-size1);
  line-height: var(--jp-private-toolbar-height);
  height: 100%;
}

/* Toolbar buttons */

/* This is the div we use to wrap the react component into a Widget */
div.jp-ToolbarButton {
  color: transparent;
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0;
  margin: 0;
}

button.jp-ToolbarButtonComponent {
  background: var(--jp-layout-color1);
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0 6px;
  margin: 0;
  height: 24px;
  border-radius: var(--jp-border-radius);
  display: flex;
  align-items: center;
  text-align: center;
  font-size: 14px;
  min-width: unset;
  min-height: unset;
}

button.jp-ToolbarButtonComponent:disabled {
  opacity: 0.4;
}

button.jp-ToolbarButtonComponent > span {
  padding: 0;
  flex: 0 0 auto;
}

button.jp-ToolbarButtonComponent .jp-ToolbarButtonComponent-label {
  font-size: var(--jp-ui-font-size1);
  line-height: 100%;
  padding-left: 2px;
  color: var(--jp-ui-font-color1);
  font-family: var(--jp-ui-font-family);
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar.jp-Toolbar-micro {
  padding: 0;
  min-height: 0;
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar {
  border: none;
  box-shadow: none;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-WindowedPanel-outer {
  position: relative;
  overflow-y: auto;
}

.jp-WindowedPanel-inner {
  position: relative;
}

.jp-WindowedPanel-window {
  position: absolute;
  left: 0;
  right: 0;
  overflow: visible;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Sibling imports */

body {
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
}

/* Disable native link decoration styles everywhere outside of dialog boxes */
a {
  text-decoration: unset;
  color: unset;
}

a:hover {
  text-decoration: unset;
  color: unset;
}

/* Accessibility for links inside dialog box text */
.jp-Dialog-content a {
  text-decoration: revert;
  color: var(--jp-content-link-color);
}

.jp-Dialog-content a:hover {
  text-decoration: revert;
}

/* Styles for ui-components */
.jp-Button {
  color: var(--jp-ui-font-color2);
  border-radius: var(--jp-border-radius);
  padding: 0 12px;
  font-size: var(--jp-ui-font-size1);

  /* Copy from blueprint 3 */
  display: inline-flex;
  flex-direction: row;
  border: none;
  cursor: pointer;
  align-items: center;
  justify-content: center;
  text-align: left;
  vertical-align: middle;
  min-height: 30px;
  min-width: 30px;
}

.jp-Button:disabled {
  cursor: not-allowed;
}

.jp-Button:empty {
  padding: 0 !important;
}

.jp-Button.jp-mod-small {
  min-height: 24px;
  min-width: 24px;
  font-size: 12px;
  padding: 0 7px;
}

/* Use our own theme for hover styles */
.jp-Button.jp-mod-minimal:hover {
  background-color: var(--jp-layout-color2);
}

.jp-Button.jp-mod-minimal {
  background: none;
}

.jp-InputGroup {
  display: block;
  position: relative;
}

.jp-InputGroup input {
  box-sizing: border-box;
  border: none;
  border-radius: 0;
  background-color: transparent;
  color: var(--jp-ui-font-color0);
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
  padding-bottom: 0;
  padding-top: 0;
  padding-left: 10px;
  padding-right: 28px;
  position: relative;
  width: 100%;
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
  font-size: 14px;
  font-weight: 400;
  height: 30px;
  line-height: 30px;
  outline: none;
  vertical-align: middle;
}

.jp-InputGroup input:focus {
  box-shadow: inset 0 0 0 var(--jp-border-width)
      var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-InputGroup input:disabled {
  cursor: not-allowed;
  resize: block;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
}

.jp-InputGroup input:disabled ~ span {
  cursor: not-allowed;
  color: var(--jp-ui-font-color2);
}

.jp-InputGroup input::placeholder,
input::placeholder {
  color: var(--jp-ui-font-color2);
}

.jp-InputGroupAction {
  position: absolute;
  bottom: 1px;
  right: 0;
  padding: 6px;
}

.jp-HTMLSelect.jp-DefaultStyle select {
  background-color: initial;
  border: none;
  border-radius: 0;
  box-shadow: none;
  color: var(--jp-ui-font-color0);
  display: block;
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  height: 24px;
  line-height: 14px;
  padding: 0 25px 0 10px;
  text-align: left;
  -moz-appearance: none;
  -webkit-appearance: none;
}

.jp-HTMLSelect.jp-DefaultStyle select:disabled {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
  cursor: not-allowed;
  resize: block;
}

.jp-HTMLSelect.jp-DefaultStyle select:disabled ~ span {
  cursor: not-allowed;
}

/* Use our own theme for hover and option styles */
/* stylelint-disable-next-line selector-max-type */
.jp-HTMLSelect.jp-DefaultStyle select:hover,
.jp-HTMLSelect.jp-DefaultStyle select > option {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color0);
}

select {
  box-sizing: border-box;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-StatusBar-Widget {
  display: flex;
  align-items: center;
  background: var(--jp-layout-color2);
  min-height: var(--jp-statusbar-height);
  justify-content: space-between;
  padding: 0 10px;
}

.jp-StatusBar-Left {
  display: flex;
  align-items: center;
  flex-direction: row;
}

.jp-StatusBar-Middle {
  display: flex;
  align-items: center;
}

.jp-StatusBar-Right {
  display: flex;
  align-items: center;
  flex-direction: row-reverse;
}

.jp-StatusBar-Item {
  max-height: var(--jp-statusbar-height);
  margin: 0 2px;
  height: var(--jp-statusbar-height);
  white-space: nowrap;
  text-overflow: ellipsis;
  color: var(--jp-ui-font-color1);
  padding: 0 6px;
}

.jp-mod-highlighted:hover {
  background-color: var(--jp-layout-color3);
}

.jp-mod-clicked {
  background-color: var(--jp-brand-color1);
}

.jp-mod-clicked:hover {
  background-color: var(--jp-brand-color0);
}

.jp-mod-clicked .jp-StatusBar-TextItem {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-StatusBar-HoverItem {
  box-shadow: '0px 4px 4px rgba(0, 0, 0, 0.25)';
}

.jp-StatusBar-TextItem {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  line-height: 24px;
  color: var(--jp-ui-font-color1);
}

.jp-StatusBar-GroupItem {
  display: flex;
  align-items: center;
  flex-direction: row;
}

.jp-Statusbar-ProgressCircle svg {
  display: block;
  margin: 0 auto;
  width: 16px;
  height: 24px;
  align-self: normal;
}

.jp-Statusbar-ProgressCircle path {
  fill: var(--jp-inverse-layout-color3);
}

.jp-Statusbar-ProgressBar-progress-bar {
  height: 10px;
  width: 100px;
  border: solid 0.25px var(--jp-brand-color2);
  border-radius: 3px;
  overflow: hidden;
  align-self: center;
}

.jp-Statusbar-ProgressBar-progress-bar > div {
  background-color: var(--jp-brand-color2);
  background-image: linear-gradient(
    -45deg,
    rgba(255, 255, 255, 0.2) 25%,
    transparent 25%,
    transparent 50%,
    rgba(255, 255, 255, 0.2) 50%,
    rgba(255, 255, 255, 0.2) 75%,
    transparent 75%,
    transparent
  );
  background-size: 40px 40px;
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 14px;
  color: #fff;
  text-align: center;
  animation: jp-Statusbar-ExecutionTime-progress-bar 2s linear infinite;
}

.jp-Statusbar-ProgressBar-progress-bar p {
  color: var(--jp-ui-font-color1);
  font-family: var(--jp-ui-font-family);
  font-size: var(--jp-ui-font-size1);
  line-height: 10px;
  width: 100px;
}

@keyframes jp-Statusbar-ExecutionTime-progress-bar {
  0% {
    background-position: 0 0;
  }

  100% {
    background-position: 40px 40px;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-commandpalette-search-height: 28px;
}

/*-----------------------------------------------------------------------------
| Overall styles
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  padding-bottom: 0;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Modal variant
|----------------------------------------------------------------------------*/

.jp-ModalCommandPalette {
  position: absolute;
  z-index: 10000;
  top: 38px;
  left: 30%;
  margin: 0;
  padding: 4px;
  width: 40%;
  box-shadow: var(--jp-elevation-z4);
  border-radius: 4px;
  background: var(--jp-layout-color0);
}

.jp-ModalCommandPalette .lm-CommandPalette {
  max-height: 40vh;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-close-icon::after {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-header {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-item {
  margin-left: 4px;
  margin-right: 4px;
}

.jp-ModalCommandPalette
  .lm-CommandPalette
  .lm-CommandPalette-item.lm-mod-disabled {
  display: none;
}

/*-----------------------------------------------------------------------------
| Search
|----------------------------------------------------------------------------*/

.lm-CommandPalette-search {
  padding: 4px;
  background-color: var(--jp-layout-color1);
  z-index: 2;
}

.lm-CommandPalette-wrapper {
  overflow: overlay;
  padding: 0 9px;
  background-color: var(--jp-input-active-background);
  height: 30px;
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.lm-CommandPalette.lm-mod-focused .lm-CommandPalette-wrapper {
  box-shadow: inset 0 0 0 1px var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-SearchIconGroup {
  color: white;
  background-color: var(--jp-brand-color1);
  position: absolute;
  top: 4px;
  right: 4px;
  padding: 5px 5px 1px;
}

.jp-SearchIconGroup svg {
  height: 20px;
  width: 20px;
}

.jp-SearchIconGroup .jp-icon3[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-input {
  background: transparent;
  width: calc(100% - 18px);
  float: left;
  border: none;
  outline: none;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  line-height: var(--jp-private-commandpalette-search-height);
}

.lm-CommandPalette-input::-webkit-input-placeholder,
.lm-CommandPalette-input::-moz-placeholder,
.lm-CommandPalette-input:-ms-input-placeholder {
  color: var(--jp-ui-font-color2);
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Results
|----------------------------------------------------------------------------*/

.lm-CommandPalette-header:first-child {
  margin-top: 0;
}

.lm-CommandPalette-header {
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin-top: 8px;
  padding: 8px 0 8px 12px;
  text-transform: uppercase;
}

.lm-CommandPalette-header.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-header > mark {
  background-color: transparent;
  font-weight: bold;
  color: var(--jp-ui-font-color1);
}

.lm-CommandPalette-item {
  padding: 4px 12px 4px 4px;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
  display: flex;
}

.lm-CommandPalette-item.lm-mod-disabled {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item.lm-mod-active {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active .jp-icon-selectable[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item:hover:not(.lm-mod-active):not(.lm-mod-disabled) {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-itemContent {
  overflow: hidden;
}

.lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.lm-CommandPalette-item.lm-mod-disabled mark {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item .lm-CommandPalette-itemIcon {
  margin: 0 4px 0 0;
  position: relative;
  width: 16px;
  top: 2px;
  flex: 0 0 auto;
}

.lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
  opacity: 0.6;
}

.lm-CommandPalette-item .lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemCaption {
  display: none;
}

.lm-CommandPalette-content {
  background-color: var(--jp-layout-color1);
}

.lm-CommandPalette-content:empty::after {
  content: 'No results';
  margin: auto;
  margin-top: 20px;
  width: 100px;
  display: block;
  font-size: var(--jp-ui-font-size2);
  font-family: var(--jp-ui-font-family);
  font-weight: lighter;
}

.lm-CommandPalette-emptyMessage {
  text-align: center;
  margin-top: 24px;
  line-height: 1.32;
  padding: 0 8px;
  color: var(--jp-content-font-color3);
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Dialog {
  position: absolute;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0;
  left: 0;
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-dialog-background);
}

.jp-Dialog-content {
  display: flex;
  flex-direction: column;
  margin-left: auto;
  margin-right: auto;
  background: var(--jp-layout-color1);
  padding: 24px 24px 12px;
  min-width: 300px;
  min-height: 150px;
  max-width: 1000px;
  max-height: 500px;
  box-sizing: border-box;
  box-shadow: var(--jp-elevation-z20);
  word-wrap: break-word;
  border-radius: var(--jp-border-radius);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
  resize: both;
}

.jp-Dialog-content.jp-Dialog-content-small {
  max-width: 500px;
}

.jp-Dialog-button {
  overflow: visible;
}

button.jp-Dialog-button:focus {
  outline: 1px solid var(--jp-brand-color1);
  outline-offset: 4px;
  -moz-outline-radius: 0;
}

button.jp-Dialog-button:focus::-moz-focus-inner {
  border: 0;
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-accept:focus,
button.jp-Dialog-button.jp-mod-styled.jp-mod-warn:focus,
button.jp-Dialog-button.jp-mod-styled.jp-mod-reject:focus {
  outline-offset: 4px;
  -moz-outline-radius: 0;
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-accept:focus {
  outline: 1px solid var(--jp-accept-color-normal, var(--jp-brand-color1));
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-warn:focus {
  outline: 1px solid var(--jp-warn-color-normal, var(--jp-error-color1));
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-reject:focus {
  outline: 1px solid var(--jp-reject-color-normal, var(--md-grey-600));
}

button.jp-Dialog-close-button {
  padding: 0;
  height: 100%;
  min-width: unset;
  min-height: unset;
}

.jp-Dialog-header {
  display: flex;
  justify-content: space-between;
  flex: 0 0 auto;
  padding-bottom: 12px;
  font-size: var(--jp-ui-font-size3);
  font-weight: 400;
  color: var(--jp-ui-font-color1);
}

.jp-Dialog-body {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  font-size: var(--jp-ui-font-size1);
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

.jp-Dialog-footer {
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  align-items: center;
  flex: 0 0 auto;
  margin-left: -12px;
  margin-right: -12px;
  padding: 12px;
}

.jp-Dialog-checkbox {
  padding-right: 5px;
}

.jp-Dialog-checkbox > input:focus-visible {
  outline: 1px solid var(--jp-input-active-border-color);
  outline-offset: 1px;
}

.jp-Dialog-spacer {
  flex: 1 1 auto;
}

.jp-Dialog-title {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-Dialog-body > .jp-select-wrapper {
  width: 100%;
}

.jp-Dialog-body > button {
  padding: 0 16px;
}

.jp-Dialog-body > label {
  line-height: 1.4;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-button.jp-mod-styled:not(:last-child) {
  margin-right: 12px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-Input-Boolean-Dialog {
  flex-direction: row-reverse;
  align-items: end;
  width: 100%;
}

.jp-Input-Boolean-Dialog > label {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MainAreaWidget > :focus {
  outline: none;
}

.jp-MainAreaWidget .jp-MainAreaWidget-error {
  padding: 6px;
}

.jp-MainAreaWidget .jp-MainAreaWidget-error > pre {
  width: auto;
  padding: 10px;
  background: var(--jp-error-color3);
  border: var(--jp-border-width) solid var(--jp-error-color1);
  border-radius: var(--jp-border-radius);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  white-space: pre-wrap;
  word-wrap: break-word;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/**
 * google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 */
:root {
  --md-red-50: #ffebee;
  --md-red-100: #ffcdd2;
  --md-red-200: #ef9a9a;
  --md-red-300: #e57373;
  --md-red-400: #ef5350;
  --md-red-500: #f44336;
  --md-red-600: #e53935;
  --md-red-700: #d32f2f;
  --md-red-800: #c62828;
  --md-red-900: #b71c1c;
  --md-red-A100: #ff8a80;
  --md-red-A200: #ff5252;
  --md-red-A400: #ff1744;
  --md-red-A700: #d50000;
  --md-pink-50: #fce4ec;
  --md-pink-100: #f8bbd0;
  --md-pink-200: #f48fb1;
  --md-pink-300: #f06292;
  --md-pink-400: #ec407a;
  --md-pink-500: #e91e63;
  --md-pink-600: #d81b60;
  --md-pink-700: #c2185b;
  --md-pink-800: #ad1457;
  --md-pink-900: #880e4f;
  --md-pink-A100: #ff80ab;
  --md-pink-A200: #ff4081;
  --md-pink-A400: #f50057;
  --md-pink-A700: #c51162;
  --md-purple-50: #f3e5f5;
  --md-purple-100: #e1bee7;
  --md-purple-200: #ce93d8;
  --md-purple-300: #ba68c8;
  --md-purple-400: #ab47bc;
  --md-purple-500: #9c27b0;
  --md-purple-600: #8e24aa;
  --md-purple-700: #7b1fa2;
  --md-purple-800: #6a1b9a;
  --md-purple-900: #4a148c;
  --md-purple-A100: #ea80fc;
  --md-purple-A200: #e040fb;
  --md-purple-A400: #d500f9;
  --md-purple-A700: #a0f;
  --md-deep-purple-50: #ede7f6;
  --md-deep-purple-100: #d1c4e9;
  --md-deep-purple-200: #b39ddb;
  --md-deep-purple-300: #9575cd;
  --md-deep-purple-400: #7e57c2;
  --md-deep-purple-500: #673ab7;
  --md-deep-purple-600: #5e35b1;
  --md-deep-purple-700: #512da8;
  --md-deep-purple-800: #4527a0;
  --md-deep-purple-900: #311b92;
  --md-deep-purple-A100: #b388ff;
  --md-deep-purple-A200: #7c4dff;
  --md-deep-purple-A400: #651fff;
  --md-deep-purple-A700: #6200ea;
  --md-indigo-50: #e8eaf6;
  --md-indigo-100: #c5cae9;
  --md-indigo-200: #9fa8da;
  --md-indigo-300: #7986cb;
  --md-indigo-400: #5c6bc0;
  --md-indigo-500: #3f51b5;
  --md-indigo-600: #3949ab;
  --md-indigo-700: #303f9f;
  --md-indigo-800: #283593;
  --md-indigo-900: #1a237e;
  --md-indigo-A100: #8c9eff;
  --md-indigo-A200: #536dfe;
  --md-indigo-A400: #3d5afe;
  --md-indigo-A700: #304ffe;
  --md-blue-50: #e3f2fd;
  --md-blue-100: #bbdefb;
  --md-blue-200: #90caf9;
  --md-blue-300: #64b5f6;
  --md-blue-400: #42a5f5;
  --md-blue-500: #2196f3;
  --md-blue-600: #1e88e5;
  --md-blue-700: #1976d2;
  --md-blue-800: #1565c0;
  --md-blue-900: #0d47a1;
  --md-blue-A100: #82b1ff;
  --md-blue-A200: #448aff;
  --md-blue-A400: #2979ff;
  --md-blue-A700: #2962ff;
  --md-light-blue-50: #e1f5fe;
  --md-light-blue-100: #b3e5fc;
  --md-light-blue-200: #81d4fa;
  --md-light-blue-300: #4fc3f7;
  --md-light-blue-400: #29b6f6;
  --md-light-blue-500: #03a9f4;
  --md-light-blue-600: #039be5;
  --md-light-blue-700: #0288d1;
  --md-light-blue-800: #0277bd;
  --md-light-blue-900: #01579b;
  --md-light-blue-A100: #80d8ff;
  --md-light-blue-A200: #40c4ff;
  --md-light-blue-A400: #00b0ff;
  --md-light-blue-A700: #0091ea;
  --md-cyan-50: #e0f7fa;
  --md-cyan-100: #b2ebf2;
  --md-cyan-200: #80deea;
  --md-cyan-300: #4dd0e1;
  --md-cyan-400: #26c6da;
  --md-cyan-500: #00bcd4;
  --md-cyan-600: #00acc1;
  --md-cyan-700: #0097a7;
  --md-cyan-800: #00838f;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84ffff;
  --md-cyan-A200: #18ffff;
  --md-cyan-A400: #00e5ff;
  --md-cyan-A700: #00b8d4;
  --md-teal-50: #e0f2f1;
  --md-teal-100: #b2dfdb;
  --md-teal-200: #80cbc4;
  --md-teal-300: #4db6ac;
  --md-teal-400: #26a69a;
  --md-teal-500: #009688;
  --md-teal-600: #00897b;
  --md-teal-700: #00796b;
  --md-teal-800: #00695c;
  --md-teal-900: #004d40;
  --md-teal-A100: #a7ffeb;
  --md-teal-A200: #64ffda;
  --md-teal-A400: #1de9b6;
  --md-teal-A700: #00bfa5;
  --md-green-50: #e8f5e9;
  --md-green-100: #c8e6c9;
  --md-green-200: #a5d6a7;
  --md-green-300: #81c784;
  --md-green-400: #66bb6a;
  --md-green-500: #4caf50;
  --md-green-600: #43a047;
  --md-green-700: #388e3c;
  --md-green-800: #2e7d32;
  --md-green-900: #1b5e20;
  --md-green-A100: #b9f6ca;
  --md-green-A200: #69f0ae;
  --md-green-A400: #00e676;
  --md-green-A700: #00c853;
  --md-light-green-50: #f1f8e9;
  --md-light-green-100: #dcedc8;
  --md-light-green-200: #c5e1a5;
  --md-light-green-300: #aed581;
  --md-light-green-400: #9ccc65;
  --md-light-green-500: #8bc34a;
  --md-light-green-600: #7cb342;
  --md-light-green-700: #689f38;
  --md-light-green-800: #558b2f;
  --md-light-green-900: #33691e;
  --md-light-green-A100: #ccff90;
  --md-light-green-A200: #b2ff59;
  --md-light-green-A400: #76ff03;
  --md-light-green-A700: #64dd17;
  --md-lime-50: #f9fbe7;
  --md-lime-100: #f0f4c3;
  --md-lime-200: #e6ee9c;
  --md-lime-300: #dce775;
  --md-lime-400: #d4e157;
  --md-lime-500: #cddc39;
  --md-lime-600: #c0ca33;
  --md-lime-700: #afb42b;
  --md-lime-800: #9e9d24;
  --md-lime-900: #827717;
  --md-lime-A100: #f4ff81;
  --md-lime-A200: #eeff41;
  --md-lime-A400: #c6ff00;
  --md-lime-A700: #aeea00;
  --md-yellow-50: #fffde7;
  --md-yellow-100: #fff9c4;
  --md-yellow-200: #fff59d;
  --md-yellow-300: #fff176;
  --md-yellow-400: #ffee58;
  --md-yellow-500: #ffeb3b;
  --md-yellow-600: #fdd835;
  --md-yellow-700: #fbc02d;
  --md-yellow-800: #f9a825;
  --md-yellow-900: #f57f17;
  --md-yellow-A100: #ffff8d;
  --md-yellow-A200: #ff0;
  --md-yellow-A400: #ffea00;
  --md-yellow-A700: #ffd600;
  --md-amber-50: #fff8e1;
  --md-amber-100: #ffecb3;
  --md-amber-200: #ffe082;
  --md-amber-300: #ffd54f;
  --md-amber-400: #ffca28;
  --md-amber-500: #ffc107;
  --md-amber-600: #ffb300;
  --md-amber-700: #ffa000;
  --md-amber-800: #ff8f00;
  --md-amber-900: #ff6f00;
  --md-amber-A100: #ffe57f;
  --md-amber-A200: #ffd740;
  --md-amber-A400: #ffc400;
  --md-amber-A700: #ffab00;
  --md-orange-50: #fff3e0;
  --md-orange-100: #ffe0b2;
  --md-orange-200: #ffcc80;
  --md-orange-300: #ffb74d;
  --md-orange-400: #ffa726;
  --md-orange-500: #ff9800;
  --md-orange-600: #fb8c00;
  --md-orange-700: #f57c00;
  --md-orange-800: #ef6c00;
  --md-orange-900: #e65100;
  --md-orange-A100: #ffd180;
  --md-orange-A200: #ffab40;
  --md-orange-A400: #ff9100;
  --md-orange-A700: #ff6d00;
  --md-deep-orange-50: #fbe9e7;
  --md-deep-orange-100: #ffccbc;
  --md-deep-orange-200: #ffab91;
  --md-deep-orange-300: #ff8a65;
  --md-deep-orange-400: #ff7043;
  --md-deep-orange-500: #ff5722;
  --md-deep-orange-600: #f4511e;
  --md-deep-orange-700: #e64a19;
  --md-deep-orange-800: #d84315;
  --md-deep-orange-900: #bf360c;
  --md-deep-orange-A100: #ff9e80;
  --md-deep-orange-A200: #ff6e40;
  --md-deep-orange-A400: #ff3d00;
  --md-deep-orange-A700: #dd2c00;
  --md-brown-50: #efebe9;
  --md-brown-100: #d7ccc8;
  --md-brown-200: #bcaaa4;
  --md-brown-300: #a1887f;
  --md-brown-400: #8d6e63;
  --md-brown-500: #795548;
  --md-brown-600: #6d4c41;
  --md-brown-700: #5d4037;
  --md-brown-800: #4e342e;
  --md-brown-900: #3e2723;
  --md-grey-50: #fafafa;
  --md-grey-100: #f5f5f5;
  --md-grey-200: #eee;
  --md-grey-300: #e0e0e0;
  --md-grey-400: #bdbdbd;
  --md-grey-500: #9e9e9e;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;
  --md-blue-grey-50: #eceff1;
  --md-blue-grey-100: #cfd8dc;
  --md-blue-grey-200: #b0bec5;
  --md-blue-grey-300: #90a4ae;
  --md-blue-grey-400: #78909c;
  --md-blue-grey-500: #607d8b;
  --md-blue-grey-600: #546e7a;
  --md-blue-grey-700: #455a64;
  --md-blue-grey-800: #37474f;
  --md-blue-grey-900: #263238;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| RenderedText
|----------------------------------------------------------------------------*/

:root {
  /* This is the padding value to fill the gaps between lines containing spans with background color. */
  --jp-private-code-span-padding: calc(
    (var(--jp-code-line-height) - 1) * var(--jp-code-font-size) / 2
  );
}

.jp-RenderedText {
  text-align: left;
  padding-left: var(--jp-code-padding);
  line-height: var(--jp-code-line-height);
  font-family: var(--jp-code-font-family);
}

.jp-RenderedText pre,
.jp-RenderedJavaScript pre,
.jp-RenderedHTMLCommon pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
  border: none;
  margin: 0;
  padding: 0;
}

.jp-RenderedText pre a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedText pre a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedText pre a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* console foregrounds and backgrounds */
.jp-RenderedText pre .ansi-black-fg {
  color: #3e424d;
}

.jp-RenderedText pre .ansi-red-fg {
  color: #e75c58;
}

.jp-RenderedText pre .ansi-green-fg {
  color: #00a250;
}

.jp-RenderedText pre .ansi-yellow-fg {
  color: #ddb62b;
}

.jp-RenderedText pre .ansi-blue-fg {
  color: #208ffb;
}

.jp-RenderedText pre .ansi-magenta-fg {
  color: #d160c4;
}

.jp-RenderedText pre .ansi-cyan-fg {
  color: #60c6c8;
}

.jp-RenderedText pre .ansi-white-fg {
  color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-bg {
  background-color: #3e424d;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-red-bg {
  background-color: #e75c58;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-green-bg {
  background-color: #00a250;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-yellow-bg {
  background-color: #ddb62b;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-blue-bg {
  background-color: #208ffb;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-magenta-bg {
  background-color: #d160c4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-cyan-bg {
  background-color: #60c6c8;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-white-bg {
  background-color: #c5c1b4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-black-intense-fg {
  color: #282c36;
}

.jp-RenderedText pre .ansi-red-intense-fg {
  color: #b22b31;
}

.jp-RenderedText pre .ansi-green-intense-fg {
  color: #007427;
}

.jp-RenderedText pre .ansi-yellow-intense-fg {
  color: #b27d12;
}

.jp-RenderedText pre .ansi-blue-intense-fg {
  color: #0065ca;
}

.jp-RenderedText pre .ansi-magenta-intense-fg {
  color: #a03196;
}

.jp-RenderedText pre .ansi-cyan-intense-fg {
  color: #258f8f;
}

.jp-RenderedText pre .ansi-white-intense-fg {
  color: #a1a6b2;
}

.jp-RenderedText pre .ansi-black-intense-bg {
  background-color: #282c36;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-red-intense-bg {
  background-color: #b22b31;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-green-intense-bg {
  background-color: #007427;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-yellow-intense-bg {
  background-color: #b27d12;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-blue-intense-bg {
  background-color: #0065ca;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-magenta-intense-bg {
  background-color: #a03196;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-cyan-intense-bg {
  background-color: #258f8f;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-white-intense-bg {
  background-color: #a1a6b2;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-default-inverse-fg {
  color: var(--jp-ui-inverse-font-color0);
}

.jp-RenderedText pre .ansi-default-inverse-bg {
  background-color: var(--jp-inverse-layout-color0);
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-bold {
  font-weight: bold;
}

.jp-RenderedText pre .ansi-underline {
  text-decoration: underline;
}

.jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
  background: var(--jp-rendermime-error-background);
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| RenderedLatex
|----------------------------------------------------------------------------*/

.jp-RenderedLatex {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
}

/* Left-justify outputs.*/
.jp-OutputArea-output.jp-RenderedLatex {
  padding: var(--jp-code-padding);
  text-align: left;
}

/*-----------------------------------------------------------------------------
| RenderedHTML
|----------------------------------------------------------------------------*/

.jp-RenderedHTMLCommon {
  color: var(--jp-content-font-color1);
  font-family: var(--jp-content-font-family);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);

  /* Give a bit more R padding on Markdown text to keep line lengths reasonable */
  padding-right: 20px;
}

.jp-RenderedHTMLCommon em {
  font-style: italic;
}

.jp-RenderedHTMLCommon strong {
  font-weight: bold;
}

.jp-RenderedHTMLCommon u {
  text-decoration: underline;
}

.jp-RenderedHTMLCommon a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* Headings */

.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4,
.jp-RenderedHTMLCommon h5,
.jp-RenderedHTMLCommon h6 {
  line-height: var(--jp-content-heading-line-height);
  font-weight: var(--jp-content-heading-font-weight);
  font-style: normal;
  margin: var(--jp-content-heading-margin-top) 0
    var(--jp-content-heading-margin-bottom) 0;
}

.jp-RenderedHTMLCommon h1:first-child,
.jp-RenderedHTMLCommon h2:first-child,
.jp-RenderedHTMLCommon h3:first-child,
.jp-RenderedHTMLCommon h4:first-child,
.jp-RenderedHTMLCommon h5:first-child,
.jp-RenderedHTMLCommon h6:first-child {
  margin-top: calc(0.5 * var(--jp-content-heading-margin-top));
}

.jp-RenderedHTMLCommon h1:last-child,
.jp-RenderedHTMLCommon h2:last-child,
.jp-RenderedHTMLCommon h3:last-child,
.jp-RenderedHTMLCommon h4:last-child,
.jp-RenderedHTMLCommon h5:last-child,
.jp-RenderedHTMLCommon h6:last-child {
  margin-bottom: calc(0.5 * var(--jp-content-heading-margin-bottom));
}

.jp-RenderedHTMLCommon h1 {
  font-size: var(--jp-content-font-size5);
}

.jp-RenderedHTMLCommon h2 {
  font-size: var(--jp-content-font-size4);
}

.jp-RenderedHTMLCommon h3 {
  font-size: var(--jp-content-font-size3);
}

.jp-RenderedHTMLCommon h4 {
  font-size: var(--jp-content-font-size2);
}

.jp-RenderedHTMLCommon h5 {
  font-size: var(--jp-content-font-size1);
}

.jp-RenderedHTMLCommon h6 {
  font-size: var(--jp-content-font-size0);
}

/* Lists */

/* stylelint-disable selector-max-type, selector-max-compound-selectors */

.jp-RenderedHTMLCommon ul:not(.list-inline),
.jp-RenderedHTMLCommon ol:not(.list-inline) {
  padding-left: 2em;
}

.jp-RenderedHTMLCommon ul {
  list-style: disc;
}

.jp-RenderedHTMLCommon ul ul {
  list-style: square;
}

.jp-RenderedHTMLCommon ul ul ul {
  list-style: circle;
}

.jp-RenderedHTMLCommon ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol ol {
  list-style: upper-alpha;
}

.jp-RenderedHTMLCommon ol ol ol {
  list-style: lower-alpha;
}

.jp-RenderedHTMLCommon ol ol ol ol {
  list-style: lower-roman;
}

.jp-RenderedHTMLCommon ol ol ol ol ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol,
.jp-RenderedHTMLCommon ul {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon ul ul,
.jp-RenderedHTMLCommon ul ol,
.jp-RenderedHTMLCommon ol ul,
.jp-RenderedHTMLCommon ol ol {
  margin-bottom: 0;
}

/* stylelint-enable selector-max-type, selector-max-compound-selectors */

.jp-RenderedHTMLCommon hr {
  color: var(--jp-border-color2);
  background-color: var(--jp-border-color1);
  margin-top: 1em;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon > pre {
  margin: 1.5em 2em;
}

.jp-RenderedHTMLCommon pre,
.jp-RenderedHTMLCommon code {
  border: 0;
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  line-height: var(--jp-code-line-height);
  padding: 0;
  white-space: pre-wrap;
}

.jp-RenderedHTMLCommon :not(pre) > code {
  background-color: var(--jp-layout-color2);
  padding: 1px 5px;
}

/* Tables */

.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
  border-spacing: 0;
  border: none;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  table-layout: fixed;
  margin-left: auto;
  margin-bottom: 1em;
  margin-right: auto;
}

.jp-RenderedHTMLCommon thead {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  vertical-align: bottom;
}

.jp-RenderedHTMLCommon td,
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon tr {
  vertical-align: middle;
  padding: 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}

.jp-RenderedMarkdown.jp-RenderedHTMLCommon td,
.jp-RenderedMarkdown.jp-RenderedHTMLCommon th {
  max-width: none;
}

:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
  text-align: right;
}

.jp-RenderedHTMLCommon th {
  font-weight: bold;
}

.jp-RenderedHTMLCommon tbody tr:nth-child(odd) {
  background: var(--jp-layout-color0);
}

.jp-RenderedHTMLCommon tbody tr:nth-child(even) {
  background: var(--jp-rendermime-table-row-background);
}

.jp-RenderedHTMLCommon tbody tr:hover {
  background: var(--jp-rendermime-table-row-hover-background);
}

.jp-RenderedHTMLCommon p {
  text-align: left;
  margin: 0;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon img {
  -moz-force-broken-image-icon: 1;
}

/* Restrict to direct children as other images could be nested in other content. */
.jp-RenderedHTMLCommon > img {
  display: block;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1em;
}

/* Change color behind transparent images if they need it... */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-light-background {
  background-color: var(--jp-inverse-layout-color1);
}

[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-dark-background {
  background-color: var(--jp-inverse-layout-color1);
}

.jp-RenderedHTMLCommon img,
.jp-RenderedImage img,
.jp-RenderedHTMLCommon svg,
.jp-RenderedSVG svg {
  max-width: 100%;
  height: auto;
}

.jp-RenderedHTMLCommon img.jp-mod-unconfined,
.jp-RenderedImage img.jp-mod-unconfined,
.jp-RenderedHTMLCommon svg.jp-mod-unconfined,
.jp-RenderedSVG svg.jp-mod-unconfined {
  max-width: none;
}

.jp-RenderedHTMLCommon .alert {
  padding: var(--jp-notebook-padding);
  border: var(--jp-border-width) solid transparent;
  border-radius: var(--jp-border-radius);
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon .alert-info {
  color: var(--jp-info-color0);
  background-color: var(--jp-info-color3);
  border-color: var(--jp-info-color2);
}

.jp-RenderedHTMLCommon .alert-info hr {
  border-color: var(--jp-info-color3);
}

.jp-RenderedHTMLCommon .alert-info > p:last-child,
.jp-RenderedHTMLCommon .alert-info > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-warning {
  color: var(--jp-warn-color0);
  background-color: var(--jp-warn-color3);
  border-color: var(--jp-warn-color2);
}

.jp-RenderedHTMLCommon .alert-warning hr {
  border-color: var(--jp-warn-color3);
}

.jp-RenderedHTMLCommon .alert-warning > p:last-child,
.jp-RenderedHTMLCommon .alert-warning > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-success {
  color: var(--jp-success-color0);
  background-color: var(--jp-success-color3);
  border-color: var(--jp-success-color2);
}

.jp-RenderedHTMLCommon .alert-success hr {
  border-color: var(--jp-success-color3);
}

.jp-RenderedHTMLCommon .alert-success > p:last-child,
.jp-RenderedHTMLCommon .alert-success > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-danger {
  color: var(--jp-error-color0);
  background-color: var(--jp-error-color3);
  border-color: var(--jp-error-color2);
}

.jp-RenderedHTMLCommon .alert-danger hr {
  border-color: var(--jp-error-color3);
}

.jp-RenderedHTMLCommon .alert-danger > p:last-child,
.jp-RenderedHTMLCommon .alert-danger > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon blockquote {
  margin: 1em 2em;
  padding: 0 1em;
  border-left: 5px solid var(--jp-border-color2);
}

a.jp-InternalAnchorLink {
  visibility: hidden;
  margin-left: 8px;
  color: var(--md-blue-800);
}

h1:hover .jp-InternalAnchorLink,
h2:hover .jp-InternalAnchorLink,
h3:hover .jp-InternalAnchorLink,
h4:hover .jp-InternalAnchorLink,
h5:hover .jp-InternalAnchorLink,
h6:hover .jp-InternalAnchorLink {
  visibility: visible;
}

.jp-RenderedHTMLCommon kbd {
  background-color: var(--jp-rendermime-table-row-background);
  border: 1px solid var(--jp-border-color0);
  border-bottom-color: var(--jp-border-color2);
  border-radius: 3px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
  display: inline-block;
  font-size: var(--jp-ui-font-size0);
  line-height: 1em;
  padding: 0.2em 0.5em;
}

/* Most direct children of .jp-RenderedHTMLCommon have a margin-bottom of 1.0.
 * At the bottom of cells this is a bit too much as there is also spacing
 * between cells. Going all the way to 0 gets too tight between markdown and
 * code cells.
 */
.jp-RenderedHTMLCommon > *:last-child {
  margin-bottom: 0.5em;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-cursor-backdrop {
  position: fixed;
  width: 200px;
  height: 200px;
  margin-top: -100px;
  margin-left: -100px;
  will-change: transform;
  z-index: 100;
}

.lm-mod-drag-image {
  will-change: transform;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-lineFormSearch {
  padding: 4px 12px;
  background-color: var(--jp-layout-color2);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
  font-size: var(--jp-ui-font-size1);
}

.jp-lineFormCaption {
  font-size: var(--jp-ui-font-size0);
  line-height: var(--jp-ui-font-size1);
  margin-top: 4px;
  color: var(--jp-ui-font-color0);
}

.jp-baseLineForm {
  border: none;
  border-radius: 0;
  position: absolute;
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  outline: none;
}

.jp-lineFormButtonContainer {
  top: 4px;
  right: 8px;
  height: 24px;
  padding: 0 12px;
  width: 12px;
}

.jp-lineFormButtonIcon {
  top: 0;
  right: 0;
  background-color: var(--jp-brand-color1);
  height: 100%;
  width: 100%;
  box-sizing: border-box;
  padding: 4px 6px;
}

.jp-lineFormButton {
  top: 0;
  right: 0;
  background-color: transparent;
  height: 100%;
  width: 100%;
  box-sizing: border-box;
}

.jp-lineFormWrapper {
  overflow: hidden;
  padding: 0 8px;
  border: 1px solid var(--jp-border-color0);
  background-color: var(--jp-input-active-background);
  height: 22px;
}

.jp-lineFormWrapperFocusWithin {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-lineFormInput {
  background: transparent;
  width: 200px;
  height: 100%;
  border: none;
  outline: none;
  color: var(--jp-ui-font-color0);
  line-height: 28px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-JSONEditor {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.jp-JSONEditor-host {
  flex: 1 1 auto;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0;
  background: var(--jp-layout-color0);
  min-height: 50px;
  padding: 1px;
}

.jp-JSONEditor.jp-mod-error .jp-JSONEditor-host {
  border-color: red;
  outline-color: red;
}

.jp-JSONEditor-header {
  display: flex;
  flex: 1 0 auto;
  padding: 0 0 0 12px;
}

.jp-JSONEditor-header label {
  flex: 0 0 auto;
}

.jp-JSONEditor-commitButton {
  height: 16px;
  width: 16px;
  background-size: 18px;
  background-repeat: no-repeat;
  background-position: center;
}

.jp-JSONEditor-host.jp-mod-focused {
  background-color: var(--jp-input-active-background);
  border: 1px solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

.jp-Editor.jp-mod-dropTarget {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/
.jp-DocumentSearch-input {
  border: none;
  outline: none;
  color: var(--jp-ui-font-color0);
  font-size: var(--jp-ui-font-size1);
  background-color: var(--jp-layout-color0);
  font-family: var(--jp-ui-font-family);
  padding: 2px 1px;
  resize: none;
}

.jp-DocumentSearch-overlay {
  position: absolute;
  background-color: var(--jp-toolbar-background);
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  border-left: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  top: 0;
  right: 0;
  z-index: 7;
  min-width: 405px;
  padding: 2px;
  font-size: var(--jp-ui-font-size1);

  --jp-private-document-search-button-height: 20px;
}

.jp-DocumentSearch-overlay button {
  background-color: var(--jp-toolbar-background);
  outline: 0;
}

.jp-DocumentSearch-overlay button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-overlay button:active {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-overlay-row {
  display: flex;
  align-items: center;
  margin-bottom: 2px;
}

.jp-DocumentSearch-button-content {
  display: inline-block;
  cursor: pointer;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-button-content svg {
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-input-wrapper {
  border: var(--jp-border-width) solid var(--jp-border-color0);
  display: flex;
  background-color: var(--jp-layout-color0);
  margin: 2px;
}

.jp-DocumentSearch-input-wrapper:focus-within {
  border-color: var(--jp-cell-editor-active-border-color);
}

.jp-DocumentSearch-toggle-wrapper,
.jp-DocumentSearch-button-wrapper {
  all: initial;
  overflow: hidden;
  display: inline-block;
  border: none;
  box-sizing: border-box;
}

.jp-DocumentSearch-toggle-wrapper {
  width: 14px;
  height: 14px;
}

.jp-DocumentSearch-button-wrapper {
  width: var(--jp-private-document-search-button-height);
  height: var(--jp-private-document-search-button-height);
}

.jp-DocumentSearch-toggle-wrapper:focus,
.jp-DocumentSearch-button-wrapper:focus {
  outline: var(--jp-border-width) solid
    var(--jp-cell-editor-active-border-color);
  outline-offset: -1px;
}

.jp-DocumentSearch-toggle-wrapper,
.jp-DocumentSearch-button-wrapper,
.jp-DocumentSearch-button-content:focus {
  outline: none;
}

.jp-DocumentSearch-toggle-placeholder {
  width: 5px;
}

.jp-DocumentSearch-input-button::before {
  display: block;
  padding-top: 100%;
}

.jp-DocumentSearch-input-button-off {
  opacity: var(--jp-search-toggle-off-opacity);
}

.jp-DocumentSearch-input-button-off:hover {
  opacity: var(--jp-search-toggle-hover-opacity);
}

.jp-DocumentSearch-input-button-on {
  opacity: var(--jp-search-toggle-on-opacity);
}

.jp-DocumentSearch-index-counter {
  padding-left: 10px;
  padding-right: 10px;
  user-select: none;
  min-width: 35px;
  display: inline-block;
}

.jp-DocumentSearch-up-down-wrapper {
  display: inline-block;
  padding-right: 2px;
  margin-left: auto;
  white-space: nowrap;
}

.jp-DocumentSearch-spacer {
  margin-left: auto;
}

.jp-DocumentSearch-up-down-wrapper button {
  outline: 0;
  border: none;
  width: var(--jp-private-document-search-button-height);
  height: var(--jp-private-document-search-button-height);
  vertical-align: middle;
  margin: 1px 5px 2px;
}

.jp-DocumentSearch-up-down-button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-up-down-button:active {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-filter-button {
  border-radius: var(--jp-border-radius);
}

.jp-DocumentSearch-filter-button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-filter-button-enabled {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-filter-button-enabled:hover {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-search-options {
  padding: 0 8px;
  margin-left: 3px;
  width: 100%;
  display: grid;
  justify-content: start;
  grid-template-columns: 1fr 1fr;
  align-items: center;
  justify-items: stretch;
}

.jp-DocumentSearch-search-filter-disabled {
  color: var(--jp-ui-font-color2);
}

.jp-DocumentSearch-search-filter {
  display: flex;
  align-items: center;
  user-select: none;
}

.jp-DocumentSearch-regex-error {
  color: var(--jp-error-color0);
}

.jp-DocumentSearch-replace-button-wrapper {
  overflow: hidden;
  display: inline-block;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color0);
  margin: auto 2px;
  padding: 1px 4px;
  height: calc(var(--jp-private-document-search-button-height) + 2px);
}

.jp-DocumentSearch-replace-button-wrapper:focus {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
}

.jp-DocumentSearch-replace-button {
  display: inline-block;
  text-align: center;
  cursor: pointer;
  box-sizing: border-box;
  color: var(--jp-ui-font-color1);

  /* height - 2 * (padding of wrapper) */
  line-height: calc(var(--jp-private-document-search-button-height) - 2px);
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-replace-button:focus {
  outline: none;
}

.jp-DocumentSearch-replace-wrapper-class {
  margin-left: 14px;
  display: flex;
}

.jp-DocumentSearch-replace-toggle {
  border: none;
  background-color: var(--jp-toolbar-background);
  border-radius: var(--jp-border-radius);
}

.jp-DocumentSearch-replace-toggle:hover {
  background-color: var(--jp-layout-color2);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.cm-editor {
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  border: 0;
  border-radius: 0;
  height: auto;

  /* Changed to auto to autogrow */
}

.cm-editor pre {
  padding: 0 var(--jp-code-padding);
}

.jp-CodeMirrorEditor[data-type='inline'] .cm-dialog {
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

.jp-CodeMirrorEditor {
  cursor: text;
}

/* When zoomed out 67% and 33% on a screen of 1440 width x 900 height */
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .jp-CodeMirrorEditor[data-type='inline'] .cm-cursor {
    border-left: var(--jp-code-cursor-width1) solid
      var(--jp-editor-cursor-color);
  }
}

/* When zoomed out less than 33% */
@media screen and (min-width: 4320px) {
  .jp-CodeMirrorEditor[data-type='inline'] .cm-cursor {
    border-left: var(--jp-code-cursor-width2) solid
      var(--jp-editor-cursor-color);
  }
}

.cm-editor.jp-mod-readOnly .cm-cursor {
  display: none;
}

.jp-CollaboratorCursor {
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: none;
  border-bottom: 3px solid;
  background-clip: content-box;
  margin-left: -5px;
  margin-right: -5px;
}

.cm-searching,
.cm-searching span {
  /* `.cm-searching span`: we need to override syntax highlighting */
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

.cm-searching::selection,
.cm-searching span::selection {
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

.jp-current-match > .cm-searching,
.jp-current-match > .cm-searching span,
.cm-searching > .jp-current-match,
.cm-searching > .jp-current-match span {
  background-color: var(--jp-search-selected-match-background-color);
  color: var(--jp-search-selected-match-color);
}

.jp-current-match > .cm-searching::selection,
.cm-searching > .jp-current-match::selection,
.jp-current-match > .cm-searching span::selection {
  background-color: var(--jp-search-selected-match-background-color);
  color: var(--jp-search-selected-match-color);
}

.cm-trailingspace {
  background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAsElEQVQIHQGlAFr/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA7+r3zKmT0/+pk9P/7+r3zAAAAAAAAAAABAAAAAAAAAAA6OPzM+/q9wAAAAAA6OPzMwAAAAAAAAAAAgAAAAAAAAAAGR8NiRQaCgAZIA0AGR8NiQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQyoYJ/SY80UAAAAASUVORK5CYII=);
  background-position: center left;
  background-repeat: repeat-x;
}

.jp-CollaboratorCursor-hover {
  position: absolute;
  z-index: 1;
  transform: translateX(-50%);
  color: white;
  border-radius: 3px;
  padding-left: 4px;
  padding-right: 4px;
  padding-top: 1px;
  padding-bottom: 1px;
  text-align: center;
  font-size: var(--jp-ui-font-size1);
  white-space: nowrap;
}

.jp-CodeMirror-ruler {
  border-left: 1px dashed var(--jp-border-color2);
}

/* Styles for shared cursors (remote cursor locations and selected ranges) */
.jp-CodeMirrorEditor .cm-ySelectionCaret {
  position: relative;
  border-left: 1px solid black;
  margin-left: -1px;
  margin-right: -1px;
  box-sizing: border-box;
}

.jp-CodeMirrorEditor .cm-ySelectionCaret > .cm-ySelectionInfo {
  white-space: nowrap;
  position: absolute;
  top: -1.15em;
  padding-bottom: 0.05em;
  left: -1px;
  font-size: 0.95em;
  font-family: var(--jp-ui-font-family);
  font-weight: bold;
  line-height: normal;
  user-select: none;
  color: white;
  padding-left: 2px;
  padding-right: 2px;
  z-index: 101;
  transition: opacity 0.3s ease-in-out;
}

.jp-CodeMirrorEditor .cm-ySelectionInfo {
  transition-delay: 0.7s;
  opacity: 0;
}

.jp-CodeMirrorEditor .cm-ySelectionCaret:hover > .cm-ySelectionInfo {
  opacity: 1;
  transition-delay: 0s;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MimeDocument {
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-filebrowser-button-height: 28px;
  --jp-private-filebrowser-button-width: 48px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FileBrowser .jp-SidePanel-content {
  display: flex;
  flex-direction: column;
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  flex-wrap: wrap;
  row-gap: 12px;
  border-bottom: none;
  height: auto;
  margin: 8px 12px 0;
  box-shadow: none;
  padding: 0;
  justify-content: flex-start;
}

.jp-FileBrowser-Panel {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
}

.jp-BreadCrumbs {
  flex: 0 0 auto;
  margin: 8px 12px;
}

.jp-BreadCrumbs-item {
  margin: 0 2px;
  padding: 0 2px;
  border-radius: var(--jp-border-radius);
  cursor: pointer;
}

.jp-BreadCrumbs-item:hover {
  background-color: var(--jp-layout-color2);
}

.jp-BreadCrumbs-item:first-child {
  margin-left: 0;
}

.jp-BreadCrumbs-item.jp-mod-dropTarget {
  background-color: var(--jp-brand-color2);
  opacity: 0.7;
}

/*-----------------------------------------------------------------------------
| Buttons
|----------------------------------------------------------------------------*/

.jp-FileBrowser-toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  padding-left: 0;
  padding-right: 2px;
  align-items: center;
  height: unset;
}

.jp-FileBrowser-toolbar > .jp-Toolbar-item .jp-ToolbarButtonComponent {
  width: 40px;
}

/*-----------------------------------------------------------------------------
| Other styles
|----------------------------------------------------------------------------*/

.jp-FileDialog.jp-mod-conflict input {
  color: var(--jp-error-color1);
}

.jp-FileDialog .jp-new-name-title {
  margin-top: 12px;
}

.jp-LastModified-hidden {
  display: none;
}

.jp-FileSize-hidden {
  display: none;
}

.jp-FileBrowser .lm-AccordionPanel > h3:first-child {
  display: none;
}

/*-----------------------------------------------------------------------------
| DirListing
|----------------------------------------------------------------------------*/

.jp-DirListing {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  outline: 0;
}

.jp-DirListing-header {
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  align-items: center;
  overflow: hidden;
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
}

.jp-DirListing-headerItem {
  padding: 4px 12px 2px;
  font-weight: 500;
}

.jp-DirListing-headerItem:hover {
  background: var(--jp-layout-color2);
}

.jp-DirListing-headerItem.jp-id-name {
  flex: 1 0 84px;
}

.jp-DirListing-headerItem.jp-id-modified {
  flex: 0 0 112px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-DirListing-headerItem.jp-id-filesize {
  flex: 0 0 75px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-id-narrow {
  display: none;
  flex: 0 0 5px;
  padding: 4px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
  color: var(--jp-border-color2);
}

.jp-DirListing-narrow .jp-id-narrow {
  display: block;
}

.jp-DirListing-narrow .jp-id-modified,
.jp-DirListing-narrow .jp-DirListing-itemModified {
  display: none;
}

.jp-DirListing-headerItem.jp-mod-selected {
  font-weight: 600;
}

/* increase specificity to override bundled default */
.jp-DirListing-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-content mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.jp-DirListing-content .jp-DirListing-item.jp-mod-selected mark {
  color: var(--jp-ui-inverse-font-color0);
}

/* Style the directory listing content when a user drops a file to upload */
.jp-DirListing.jp-mod-native-drop .jp-DirListing-content {
  outline: 5px dashed rgba(128, 128, 128, 0.5);
  outline-offset: -10px;
  cursor: copy;
}

.jp-DirListing-item {
  display: flex;
  flex-direction: row;
  align-items: center;
  padding: 4px 12px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-DirListing-checkboxWrapper {
  /* Increases hit area of checkbox. */
  padding: 4px;
}

.jp-DirListing-header
  .jp-DirListing-checkboxWrapper
  + .jp-DirListing-headerItem {
  padding-left: 4px;
}

.jp-DirListing-content .jp-DirListing-checkboxWrapper {
  position: relative;
  left: -4px;
  margin: -4px 0 -4px -8px;
}

.jp-DirListing-checkboxWrapper.jp-mod-visible {
  visibility: visible;
}

/* For devices that support hovering, hide checkboxes until hovered, selected...
*/
@media (hover: hover) {
  .jp-DirListing-checkboxWrapper {
    visibility: hidden;
  }

  .jp-DirListing-item:hover .jp-DirListing-checkboxWrapper,
  .jp-DirListing-item.jp-mod-selected .jp-DirListing-checkboxWrapper {
    visibility: visible;
  }
}

.jp-DirListing-item[data-is-dot] {
  opacity: 75%;
}

.jp-DirListing-item.jp-mod-selected {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.jp-DirListing-item.jp-mod-dropTarget {
  background: var(--jp-brand-color3);
}

.jp-DirListing-item:hover:not(.jp-mod-selected) {
  background: var(--jp-layout-color2);
}

.jp-DirListing-itemIcon {
  flex: 0 0 20px;
  margin-right: 4px;
}

.jp-DirListing-itemText {
  flex: 1 0 64px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  user-select: none;
}

.jp-DirListing-itemText:focus {
  outline-width: 2px;
  outline-color: var(--jp-inverse-layout-color1);
  outline-style: solid;
  outline-offset: 1px;
}

.jp-DirListing-item.jp-mod-selected .jp-DirListing-itemText:focus {
  outline-color: var(--jp-layout-color1);
}

.jp-DirListing-itemModified {
  flex: 0 0 125px;
  text-align: right;
}

.jp-DirListing-itemFileSize {
  flex: 0 0 90px;
  text-align: right;
}

.jp-DirListing-editor {
  flex: 1 0 64px;
  outline: none;
  border: none;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-item.jp-mod-running .jp-DirListing-itemIcon::before {
  color: var(--jp-success-color1);
  content: '\25CF';
  font-size: 8px;
  position: absolute;
  left: -8px;
}

.jp-DirListing-item.jp-mod-running.jp-mod-selected
  .jp-DirListing-itemIcon::before {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-DirListing-item.lm-mod-drag-image,
.jp-DirListing-item.jp-mod-selected.lm-mod-drag-image {
  font-size: var(--jp-ui-font-size1);
  padding-left: 4px;
  margin-left: 4px;
  width: 160px;
  background-color: var(--jp-ui-inverse-font-color2);
  box-shadow: var(--jp-elevation-z2);
  border-radius: 0;
  color: var(--jp-ui-font-color1);
  transform: translateX(-40%) translateY(-58%);
}

.jp-Document {
  min-width: 120px;
  min-height: 120px;
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Main OutputArea
| OutputArea has a list of Outputs
|----------------------------------------------------------------------------*/

.jp-OutputArea {
  overflow-y: auto;
}

.jp-OutputArea-child {
  display: table;
  table-layout: fixed;
  width: 100%;
  overflow: hidden;
}

.jp-OutputPrompt {
  width: var(--jp-cell-prompt-width);
  color: var(--jp-cell-outprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);

  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-OutputArea-prompt {
  display: table-cell;
  vertical-align: top;
}

.jp-OutputArea-output {
  display: table-cell;
  width: 100%;
  height: auto;
  overflow: auto;
  user-select: text;
  -moz-user-select: text;
  -webkit-user-select: text;
  -ms-user-select: text;
}

.jp-OutputArea .jp-RenderedText {
  padding-left: 1ch;
}

/**
 * Prompt overlay.
 */

.jp-OutputArea-promptOverlay {
  position: absolute;
  top: 0;
  width: var(--jp-cell-prompt-width);
  height: 100%;
  opacity: 0.5;
}

.jp-OutputArea-promptOverlay:hover {
  background: var(--jp-layout-color2);
  box-shadow: inset 0 0 1px var(--jp-inverse-layout-color0);
  cursor: zoom-out;
}

.jp-mod-outputsScrolled .jp-OutputArea-promptOverlay:hover {
  cursor: zoom-in;
}

/**
 * Isolated output.
 */
.jp-OutputArea-output.jp-mod-isolated {
  width: 100%;
  display: block;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated {
  position: relative;
}

body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/* pre */

.jp-OutputArea-output pre {
  border: none;
  margin: 0;
  padding: 0;
  overflow-x: auto;
  overflow-y: auto;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* tables */

.jp-OutputArea-output.jp-RenderedHTMLCommon table {
  margin-left: 0;
  margin-right: 0;
}

/* description lists */

.jp-OutputArea-output dl,
.jp-OutputArea-output dt,
.jp-OutputArea-output dd {
  display: block;
}

.jp-OutputArea-output dl {
  width: 100%;
  overflow: hidden;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dt {
  font-weight: bold;
  float: left;
  width: 20%;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dd {
  float: left;
  width: 80%;
  padding: 0;
  margin: 0;
}

.jp-TrimmedOutputs pre {
  background: var(--jp-layout-color3);
  font-size: calc(var(--jp-code-font-size) * 1.4);
  text-align: center;
  text-transform: uppercase;
}

/* Hide the gutter in case of
 *  - nested output areas (e.g. in the case of output widgets)
 *  - mirrored output areas
 */
.jp-OutputArea .jp-OutputArea .jp-OutputArea-prompt {
  display: none;
}

/* Hide empty lines in the output area, for instance due to cleared widgets */
.jp-OutputArea-prompt:empty {
  padding: 0;
  border: 0;
}

/*-----------------------------------------------------------------------------
| executeResult is added to any Output-result for the display of the object
| returned by a cell
|----------------------------------------------------------------------------*/

.jp-OutputArea-output.jp-OutputArea-executeResult {
  margin-left: 0;
  width: 100%;
}

/* Text output with the Out[] prompt needs a top padding to match the
 * alignment of the Out[] prompt itself.
 */
.jp-OutputArea-executeResult .jp-RenderedText.jp-OutputArea-output {
  padding-top: var(--jp-code-padding);
  border-top: var(--jp-border-width) solid transparent;
}

/*-----------------------------------------------------------------------------
| The Stdin output
|----------------------------------------------------------------------------*/

.jp-Stdin-prompt {
  color: var(--jp-content-font-color0);
  padding-right: var(--jp-code-padding);
  vertical-align: baseline;
  flex: 0 0 auto;
}

.jp-Stdin-input {
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  color: inherit;
  background-color: inherit;
  width: 42%;
  min-width: 200px;

  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;

  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0 0.25em;
  margin: 0 0.25em;
  flex: 0 0 70%;
}

.jp-Stdin-input::placeholder {
  opacity: 0;
}

.jp-Stdin-input:focus {
  box-shadow: none;
}

.jp-Stdin-input:focus::placeholder {
  opacity: 1;
}

/*-----------------------------------------------------------------------------
| Output Area View
|----------------------------------------------------------------------------*/

.jp-LinkedOutputView .jp-OutputArea {
  height: 100%;
  display: block;
}

.jp-LinkedOutputView .jp-OutputArea-output:only-child {
  height: 100%;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

@media print {
  .jp-OutputArea-child {
    break-inside: avoid-page;
  }
}

/*-----------------------------------------------------------------------------
| Mobile
|----------------------------------------------------------------------------*/
@media only screen and (max-width: 760px) {
  .jp-OutputPrompt {
    display: table-row;
    text-align: left;
  }

  .jp-OutputArea-child .jp-OutputArea-output {
    display: table-row;
    margin-left: var(--jp-notebook-padding);
  }
}

/* Trimmed outputs warning */
.jp-TrimmedOutputs > a {
  margin: 10px;
  text-decoration: none;
  cursor: pointer;
}

.jp-TrimmedOutputs > a:hover {
  text-decoration: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Table of Contents
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toc-active-width: 4px;
}

.jp-TableOfContents {
  display: flex;
  flex-direction: column;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  height: 100%;
}

.jp-TableOfContents-placeholder {
  text-align: center;
}

.jp-TableOfContents-placeholderContent {
  color: var(--jp-content-font-color2);
  padding: 8px;
}

.jp-TableOfContents-placeholderContent > h3 {
  margin-bottom: var(--jp-content-heading-margin-bottom);
}

.jp-TableOfContents .jp-SidePanel-content {
  overflow-y: auto;
}

.jp-TableOfContents-tree {
  margin: 4px;
}

.jp-TableOfContents ol {
  list-style-type: none;
}

/* stylelint-disable-next-line selector-max-type */
.jp-TableOfContents li > ol {
  /* Align left border with triangle icon center */
  padding-left: 11px;
}

.jp-TableOfContents-content {
  /* left margin for the active heading indicator */
  margin: 0 0 0 var(--jp-private-toc-active-width);
  padding: 0;
  background-color: var(--jp-layout-color1);
}

.jp-tocItem {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-tocItem-heading {
  display: flex;
  cursor: pointer;
}

.jp-tocItem-heading:hover {
  background-color: var(--jp-layout-color2);
}

.jp-tocItem-content {
  display: block;
  padding: 4px 0;
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow-x: hidden;
}

.jp-tocItem-collapser {
  height: 20px;
  margin: 2px 2px 0;
  padding: 0;
  background: none;
  border: none;
  cursor: pointer;
}

.jp-tocItem-collapser:hover {
  background-color: var(--jp-layout-color3);
}

/* Active heading indicator */

.jp-tocItem-heading::before {
  content: ' ';
  background: transparent;
  width: var(--jp-private-toc-active-width);
  height: 24px;
  position: absolute;
  left: 0;
  border-radius: var(--jp-border-radius);
}

.jp-tocItem-heading.jp-tocItem-active::before {
  background-color: var(--jp-brand-color1);
}

.jp-tocItem-heading:hover.jp-tocItem-active::before {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapser {
  flex: 0 0 var(--jp-cell-collapser-width);
  padding: 0;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
  border-radius: var(--jp-border-radius);
  opacity: 1;
}

.jp-Collapser-child {
  display: block;
  width: 100%;
  box-sizing: border-box;

  /* height: 100% doesn't work because the height of its parent is computed from content */
  position: absolute;
  top: 0;
  bottom: 0;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

/*
Hiding collapsers in print mode.

Note: input and output wrappers have "display: block" propery in print mode.
*/

@media print {
  .jp-Collapser {
    display: none;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Header/Footer
|----------------------------------------------------------------------------*/

/* Hidden by zero height by default */
.jp-CellHeader,
.jp-CellFooter {
  height: 0;
  width: 100%;
  padding: 0;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Input
|----------------------------------------------------------------------------*/

/* All input areas */
.jp-InputArea {
  display: table;
  table-layout: fixed;
  width: 100%;
  overflow: hidden;
}

.jp-InputArea-editor {
  display: table-cell;
  overflow: hidden;
  vertical-align: top;

  /* This is the non-active, default styling */
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0;
  background: var(--jp-cell-editor-background);
}

.jp-InputPrompt {
  display: table-cell;
  vertical-align: top;
  width: var(--jp-cell-prompt-width);
  color: var(--jp-cell-inprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  opacity: var(--jp-cell-prompt-opacity);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;

  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/*-----------------------------------------------------------------------------
| Mobile
|----------------------------------------------------------------------------*/
@media only screen and (max-width: 760px) {
  .jp-InputArea-editor {
    display: table-row;
    margin-left: var(--jp-notebook-padding);
  }

  .jp-InputPrompt {
    display: table-row;
    text-align: left;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Placeholder {
  display: table;
  table-layout: fixed;
  width: 100%;
}

.jp-Placeholder-prompt {
  display: table-cell;
  box-sizing: border-box;
}

.jp-Placeholder-content {
  display: table-cell;
  padding: 4px 6px;
  border: 1px solid transparent;
  border-radius: 0;
  background: none;
  box-sizing: border-box;
  cursor: pointer;
}

.jp-Placeholder-contentContainer {
  display: flex;
}

.jp-Placeholder-content:hover,
.jp-InputPlaceholder > .jp-Placeholder-content:hover {
  border-color: var(--jp-layout-color3);
}

.jp-Placeholder-content .jp-MoreHorizIcon {
  width: 32px;
  height: 16px;
  border: 1px solid transparent;
  border-radius: var(--jp-border-radius);
}

.jp-Placeholder-content .jp-MoreHorizIcon:hover {
  border: 1px solid var(--jp-border-color1);
  box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.25);
  background-color: var(--jp-layout-color0);
}

.jp-PlaceholderText {
  white-space: nowrap;
  overflow-x: hidden;
  color: var(--jp-inverse-layout-color3);
  font-family: var(--jp-code-font-family);
}

.jp-InputPlaceholder > .jp-Placeholder-content {
  border-color: var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-cell-scrolling-output-offset: 5px;
}

/*-----------------------------------------------------------------------------
| Cell
|----------------------------------------------------------------------------*/

.jp-Cell {
  padding: var(--jp-cell-padding);
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Common input/output
|----------------------------------------------------------------------------*/

.jp-Cell-inputWrapper,
.jp-Cell-outputWrapper {
  display: flex;
  flex-direction: row;
  padding: 0;
  margin: 0;

  /* Added to reveal the box-shadow on the input and output collapsers. */
  overflow: visible;
}

/* Only input/output areas inside cells */
.jp-Cell-inputArea,
.jp-Cell-outputArea {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Collapser
|----------------------------------------------------------------------------*/

/* Make the output collapser disappear when there is not output, but do so
 * in a manner that leaves it in the layout and preserves its width.
 */
.jp-Cell.jp-mod-noOutputs .jp-Cell-outputCollapser {
  border: none !important;
  background: transparent !important;
}

.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputCollapser {
  min-height: var(--jp-cell-collapser-min-height);
}

/*-----------------------------------------------------------------------------
| Output
|----------------------------------------------------------------------------*/

/* Put a space between input and output when there IS output */
.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper {
  margin-top: 5px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea {
  overflow-y: auto;
  max-height: 24em;
  margin-left: var(--jp-private-cell-scrolling-output-offset);
  resize: vertical;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea[style*='height'] {
  max-height: unset;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea::after {
  content: ' ';
  box-shadow: inset 0 0 6px 2px rgb(0 0 0 / 30%);
  width: 100%;
  height: 100%;
  position: sticky;
  bottom: 0;
  top: 0;
  margin-top: -50%;
  float: left;
  display: block;
  pointer-events: none;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-child {
  padding-top: 6px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  width: calc(
    var(--jp-cell-prompt-width) - var(--jp-private-cell-scrolling-output-offset)
  );
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-promptOverlay {
  left: calc(-1 * var(--jp-private-cell-scrolling-output-offset));
}

/*-----------------------------------------------------------------------------
| CodeCell
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| MarkdownCell
|----------------------------------------------------------------------------*/

.jp-MarkdownOutput {
  display: table-cell;
  width: 100%;
  margin-top: 0;
  margin-bottom: 0;
  padding-left: var(--jp-code-padding);
}

.jp-MarkdownOutput.jp-RenderedHTMLCommon {
  overflow: auto;
}

/* collapseHeadingButton (show always if hiddenCellsButton is _not_ shown) */
.jp-collapseHeadingButton {
  display: flex;
  min-height: var(--jp-cell-collapser-min-height);
  font-size: var(--jp-code-font-size);
  position: absolute;
  background-color: transparent;
  background-size: 25px;
  background-repeat: no-repeat;
  background-position-x: center;
  background-position-y: top;
  background-image: var(--jp-icon-caret-down);
  right: 0;
  top: 0;
  bottom: 0;
}

.jp-collapseHeadingButton.jp-mod-collapsed {
  background-image: var(--jp-icon-caret-right);
}

/*
 set the container font size to match that of content
 so that the nested collapse buttons have the right size
*/
.jp-MarkdownCell .jp-InputPrompt {
  font-size: var(--jp-content-font-size1);
}

/*
  Align collapseHeadingButton with cell top header
  The font sizes are identical to the ones in packages/rendermime/style/base.css
*/
.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='1'] {
  font-size: var(--jp-content-font-size5);
  background-position-y: calc(0.3 * var(--jp-content-font-size5));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='2'] {
  font-size: var(--jp-content-font-size4);
  background-position-y: calc(0.3 * var(--jp-content-font-size4));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='3'] {
  font-size: var(--jp-content-font-size3);
  background-position-y: calc(0.3 * var(--jp-content-font-size3));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='4'] {
  font-size: var(--jp-content-font-size2);
  background-position-y: calc(0.3 * var(--jp-content-font-size2));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='5'] {
  font-size: var(--jp-content-font-size1);
  background-position-y: top;
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='6'] {
  font-size: var(--jp-content-font-size0);
  background-position-y: top;
}

/* collapseHeadingButton (show only on (hover,active) if hiddenCellsButton is shown) */
.jp-Notebook.jp-mod-showHiddenCellsButton .jp-collapseHeadingButton {
  display: none;
}

.jp-Notebook.jp-mod-showHiddenCellsButton
  :is(.jp-MarkdownCell:hover, .jp-mod-active)
  .jp-collapseHeadingButton {
  display: flex;
}

/* showHiddenCellsButton (only show if jp-mod-showHiddenCellsButton is set, which
is a consequence of the showHiddenCellsButton option in Notebook Settings)*/
.jp-Notebook.jp-mod-showHiddenCellsButton .jp-showHiddenCellsButton {
  margin-left: calc(var(--jp-cell-prompt-width) + 2 * var(--jp-code-padding));
  margin-top: var(--jp-code-padding);
  border: 1px solid var(--jp-border-color2);
  background-color: var(--jp-border-color3) !important;
  color: var(--jp-content-font-color0) !important;
  display: flex;
}

.jp-Notebook.jp-mod-showHiddenCellsButton .jp-showHiddenCellsButton:hover {
  background-color: var(--jp-border-color2) !important;
}

.jp-showHiddenCellsButton {
  display: none;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

/*
Using block instead of flex to allow the use of the break-inside CSS property for
cell outputs.
*/

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-notebook-toolbar-padding: 2px 5px 2px 2px;
}

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-NotebookPanel-toolbar {
  padding: var(--jp-notebook-toolbar-padding);

  /* disable paint containment from lumino 2.0 default strict CSS containment */
  contain: style size !important;
}

.jp-Toolbar-item.jp-Notebook-toolbarCellType .jp-select-wrapper.jp-mod-focused {
  border: none;
  box-shadow: none;
}

.jp-Notebook-toolbarCellTypeDropdown select {
  height: 24px;
  font-size: var(--jp-ui-font-size1);
  line-height: 14px;
  border-radius: 0;
  display: block;
}

.jp-Notebook-toolbarCellTypeDropdown span {
  top: 5px !important;
}

.jp-Toolbar-responsive-popup {
  position: absolute;
  height: fit-content;
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  justify-content: flex-end;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: var(--jp-notebook-toolbar-padding);
  z-index: 1;
  right: 0;
  top: 0;
}

.jp-Toolbar > .jp-Toolbar-responsive-opener {
  margin-left: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-Notebook-ExecutionIndicator {
  position: relative;
  display: inline-block;
  height: 100%;
  z-index: 9997;
}

.jp-Notebook-ExecutionIndicator-tooltip {
  visibility: hidden;
  height: auto;
  width: max-content;
  width: -moz-max-content;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color1);
  text-align: justify;
  border-radius: 6px;
  padding: 0 5px;
  position: fixed;
  display: table;
}

.jp-Notebook-ExecutionIndicator-tooltip.up {
  transform: translateX(-50%) translateY(-100%) translateY(-32px);
}

.jp-Notebook-ExecutionIndicator-tooltip.down {
  transform: translateX(calc(-100% + 16px)) translateY(5px);
}

.jp-Notebook-ExecutionIndicator-tooltip.hidden {
  display: none;
}

.jp-Notebook-ExecutionIndicator:hover .jp-Notebook-ExecutionIndicator-tooltip {
  visibility: visible;
}

.jp-Notebook-ExecutionIndicator span {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  color: var(--jp-ui-font-color1);
  line-height: 24px;
  display: block;
}

.jp-Notebook-ExecutionIndicator-progress-bar {
  display: flex;
  justify-content: center;
  height: 100%;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*
 * Execution indicator
 */
.jp-tocItem-content::after {
  content: '';

  /* Must be identical to form a circle */
  width: 12px;
  height: 12px;
  background: none;
  border: none;
  position: absolute;
  right: 0;
}

.jp-tocItem-content[data-running='0']::after {
  border-radius: 50%;
  border: var(--jp-border-width) solid var(--jp-inverse-layout-color3);
  background: none;
}

.jp-tocItem-content[data-running='1']::after {
  border-radius: 50%;
  border: var(--jp-border-width) solid var(--jp-inverse-layout-color3);
  background-color: var(--jp-inverse-layout-color3);
}

.jp-tocItem-content[data-running='0'],
.jp-tocItem-content[data-running='1'] {
  margin-right: 12px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-Notebook-footer {
  height: 27px;
  margin-left: calc(
    var(--jp-cell-prompt-width) + var(--jp-cell-collapser-width) +
      var(--jp-cell-padding)
  );
  width: calc(
    100% -
      (
        var(--jp-cell-prompt-width) + var(--jp-cell-collapser-width) +
          var(--jp-cell-padding) + var(--jp-cell-padding)
      )
  );
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  color: var(--jp-ui-font-color3);
  margin-top: 6px;
  background: none;
  cursor: pointer;
}

.jp-Notebook-footer:focus {
  border-color: var(--jp-cell-editor-active-border-color);
}

/* For devices that support hovering, hide footer until hover */
@media (hover: hover) {
  .jp-Notebook-footer {
    opacity: 0;
  }

  .jp-Notebook-footer:focus,
  .jp-Notebook-footer:hover {
    opacity: 1;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Imports
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-side-by-side-output-size: 1fr;
  --jp-side-by-side-resized-cell: var(--jp-side-by-side-output-size);
  --jp-private-notebook-dragImage-width: 304px;
  --jp-private-notebook-dragImage-height: 36px;
  --jp-private-notebook-selected-color: var(--md-blue-400);
  --jp-private-notebook-active-color: var(--md-green-400);
}

/*-----------------------------------------------------------------------------
| Notebook
|----------------------------------------------------------------------------*/

/* stylelint-disable selector-max-class */

.jp-NotebookPanel {
  display: block;
  height: 100%;
}

.jp-NotebookPanel.jp-Document {
  min-width: 240px;
  min-height: 120px;
}

.jp-Notebook {
  padding: var(--jp-notebook-padding);
  outline: none;
  overflow: auto;
  background: var(--jp-layout-color0);
}

.jp-Notebook.jp-mod-scrollPastEnd::after {
  display: block;
  content: '';
  min-height: var(--jp-notebook-scroll-padding);
}

.jp-MainAreaWidget-ContainStrict .jp-Notebook * {
  contain: strict;
}

.jp-Notebook .jp-Cell {
  overflow: visible;
}

.jp-Notebook .jp-Cell .jp-InputPrompt {
  cursor: move;
}

/*-----------------------------------------------------------------------------
| Notebook state related styling
|
| The notebook and cells each have states, here are the possibilities:
|
| - Notebook
|   - Command
|   - Edit
| - Cell
|   - None
|   - Active (only one can be active)
|   - Selected (the cells actions are applied to)
|   - Multiselected (when multiple selected, the cursor)
|   - No outputs
|----------------------------------------------------------------------------*/

/* Command or edit modes */

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

/* cell is active */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser {
  background: var(--jp-brand-color1);
}

/* cell is dirty */
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt {
  color: var(--jp-warn-color1);
}

.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt::before {
  color: var(--jp-warn-color1);
  content: '•';
}

.jp-Notebook .jp-Cell.jp-mod-active.jp-mod-dirty .jp-Collapser {
  background: var(--jp-warn-color1);
}

/* collapser is hovered */
.jp-Notebook .jp-Cell .jp-Collapser:hover {
  box-shadow: var(--jp-elevation-z2);
  background: var(--jp-brand-color1);
  opacity: var(--jp-cell-collapser-not-active-hover-opacity);
}

/* cell is active and collapser is hovered */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser:hover {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/* Command mode */

.jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected {
  background: var(--jp-notebook-multiselected-color);
}

.jp-Notebook.jp-mod-commandMode
  .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
  background: transparent;
}

/* Edit mode */

.jp-Notebook.jp-mod-editMode .jp-Cell.jp-mod-active .jp-InputArea-editor {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-cell-editor-active-background);
}

/*-----------------------------------------------------------------------------
| Notebook drag and drop
|----------------------------------------------------------------------------*/

.jp-Notebook-cell.jp-mod-dropSource {
  opacity: 0.5;
}

.jp-Notebook-cell.jp-mod-dropTarget,
.jp-Notebook.jp-mod-commandMode
  .jp-Notebook-cell.jp-mod-active.jp-mod-selected.jp-mod-dropTarget {
  border-top-color: var(--jp-private-notebook-selected-color);
  border-top-style: solid;
  border-top-width: 2px;
}

.jp-dragImage {
  display: block;
  flex-direction: row;
  width: var(--jp-private-notebook-dragImage-width);
  height: var(--jp-private-notebook-dragImage-height);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
  overflow: visible;
}

.jp-dragImage-singlePrompt {
  box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.12);
}

.jp-dragImage .jp-dragImage-content {
  flex: 1 1 auto;
  z-index: 2;
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  line-height: var(--jp-code-line-height);
  padding: var(--jp-code-padding);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background-color);
  color: var(--jp-content-font-color3);
  text-align: left;
  margin: 4px 4px 4px 0;
}

.jp-dragImage .jp-dragImage-prompt {
  flex: 0 0 auto;
  min-width: 36px;
  color: var(--jp-cell-inprompt-font-color);
  padding: var(--jp-code-padding);
  padding-left: 12px;
  font-family: var(--jp-cell-prompt-font-family);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: 1.9;
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
}

.jp-dragImage-multipleBack {
  z-index: -1;
  position: absolute;
  height: 32px;
  width: 300px;
  top: 8px;
  left: 8px;
  background: var(--jp-layout-color2);
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.12);
}

/*-----------------------------------------------------------------------------
| Cell toolbar
|----------------------------------------------------------------------------*/

.jp-NotebookTools {
  display: block;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
    * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  overflow: auto;
}

.jp-ActiveCellTool {
  padding: 12px 0;
  display: flex;
}

.jp-ActiveCellTool-Content {
  flex: 1 1 auto;
}

.jp-ActiveCellTool .jp-ActiveCellTool-CellContent {
  background: var(--jp-cell-editor-background);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0;
  min-height: 29px;
}

.jp-ActiveCellTool .jp-InputPrompt {
  min-width: calc(var(--jp-cell-prompt-width) * 0.75);
}

.jp-ActiveCellTool-CellContent > pre {
  padding: 5px 4px;
  margin: 0;
  white-space: normal;
}

.jp-MetadataEditorTool {
  flex-direction: column;
  padding: 12px 0;
}

.jp-RankedPanel > :not(:first-child) {
  margin-top: 12px;
}

.jp-KeySelector select.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
}

.jp-KeySelector label,
.jp-MetadataEditorTool label,
.jp-NumberSetter label {
  line-height: 1.4;
}

.jp-NotebookTools .jp-select-wrapper {
  margin-top: 4px;
  margin-bottom: 0;
}

.jp-NumberSetter input {
  width: 100%;
  margin-top: 4px;
}

.jp-NotebookTools .jp-Collapse {
  margin-top: 16px;
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-Notebook {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-InputPrompt,
.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-OutputPrompt {
  flex: 0 0 110px;
}

/*-----------------------------------------------------------------------------
| Side-by-side Mode (.jp-mod-sideBySide)
|----------------------------------------------------------------------------*/
.jp-mod-sideBySide.jp-Notebook .jp-Notebook-cell {
  margin-top: 3em;
  margin-bottom: 3em;
  margin-left: 5%;
  margin-right: 5%;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell {
  display: grid;
  grid-template-columns: minmax(0, 1fr) min-content minmax(
      0,
      var(--jp-side-by-side-output-size)
    );
  grid-template-rows: auto minmax(0, 1fr) auto;
  grid-template-areas:
    'header header header'
    'input handle output'
    'footer footer footer';
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell.jp-mod-resizedCell {
  grid-template-columns: minmax(0, 1fr) min-content minmax(
      0,
      var(--jp-side-by-side-resized-cell)
    );
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellHeader {
  grid-area: header;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-Cell-inputWrapper {
  grid-area: input;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-Cell-outputWrapper {
  /* overwrite the default margin (no vertical separation needed in side by side move */
  margin-top: 0;
  grid-area: output;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellFooter {
  grid-area: footer;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellResizeHandle {
  grid-area: handle;
  user-select: none;
  display: block;
  height: 100%;
  cursor: ew-resize;
  padding: 0 var(--jp-cell-padding);
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellResizeHandle::after {
  content: '';
  display: block;
  background: var(--jp-border-color2);
  height: 100%;
  width: 5px;
}

.jp-mod-sideBySide.jp-Notebook
  .jp-CodeCell.jp-mod-resizedCell
  .jp-CellResizeHandle::after {
  background: var(--jp-border-color0);
}

.jp-CellResizeHandle {
  display: none;
}

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Cell-Placeholder {
  padding-left: 55px;
}

.jp-Cell-Placeholder-wrapper {
  background: #fff;
  border: 1px solid;
  border-color: #e5e6e9 #dfe0e4 #d0d1d5;
  border-radius: 4px;
  -webkit-border-radius: 4px;
  margin: 10px 15px;
}

.jp-Cell-Placeholder-wrapper-inner {
  padding: 15px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body {
  background-repeat: repeat;
  background-size: 50% auto;
}

.jp-Cell-Placeholder-wrapper-body div {
  background: #f6f7f8;
  background-image: -webkit-linear-gradient(
    left,
    #f6f7f8 0%,
    #edeef1 20%,
    #f6f7f8 40%,
    #f6f7f8 100%
  );
  background-repeat: no-repeat;
  background-size: 800px 104px;
  height: 104px;
  position: absolute;
  right: 15px;
  left: 15px;
  top: 15px;
}

div.jp-Cell-Placeholder-h1 {
  top: 20px;
  height: 20px;
  left: 15px;
  width: 150px;
}

div.jp-Cell-Placeholder-h2 {
  left: 15px;
  top: 50px;
  height: 10px;
  width: 100px;
}

div.jp-Cell-Placeholder-content-1,
div.jp-Cell-Placeholder-content-2,
div.jp-Cell-Placeholder-content-3 {
  left: 15px;
  right: 15px;
  height: 10px;
}

div.jp-Cell-Placeholder-content-1 {
  top: 100px;
}

div.jp-Cell-Placeholder-content-2 {
  top: 120px;
}

div.jp-Cell-Placeholder-content-3 {
  top: 140px;
}

</style>
<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
The following CSS variables define the main, public API for styling JupyterLab.
These variables should be used by all plugins wherever possible. In other
words, plugins should not define custom colors, sizes, etc unless absolutely
necessary. This enables users to change the visual theme of JupyterLab
by changing these variables.

Many variables appear in an ordered sequence (0,1,2,3). These sequences
are designed to work well together, so for example, `--jp-border-color1` should
be used with `--jp-layout-color1`. The numbers have the following meanings:

* 0: super-primary, reserved for special emphasis
* 1: primary, most important under normal situations
* 2: secondary, next most important under normal situations
* 3: tertiary, next most important under normal situations

Throughout JupyterLab, we are mostly following principles from Google's
Material Design when selecting colors. We are not, however, following
all of MD as it is not optimized for dense, information rich UIs.
*/

:root {
  /* Elevation
   *
   * We style box-shadows using Material Design's idea of elevation. These particular numbers are taken from here:
   *
   * https://github.com/material-components/material-components-web
   * https://material-components-web.appspot.com/elevation.html
   */

  --jp-shadow-base-lightness: 0;
  --jp-shadow-umbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.2
  );
  --jp-shadow-penumbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.14
  );
  --jp-shadow-ambient-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.12
  );
  --jp-elevation-z0: none;
  --jp-elevation-z1: 0 2px 1px -1px var(--jp-shadow-umbra-color),
    0 1px 1px 0 var(--jp-shadow-penumbra-color),
    0 1px 3px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z2: 0 3px 1px -2px var(--jp-shadow-umbra-color),
    0 2px 2px 0 var(--jp-shadow-penumbra-color),
    0 1px 5px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z4: 0 2px 4px -1px var(--jp-shadow-umbra-color),
    0 4px 5px 0 var(--jp-shadow-penumbra-color),
    0 1px 10px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z6: 0 3px 5px -1px var(--jp-shadow-umbra-color),
    0 6px 10px 0 var(--jp-shadow-penumbra-color),
    0 1px 18px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z8: 0 5px 5px -3px var(--jp-shadow-umbra-color),
    0 8px 10px 1px var(--jp-shadow-penumbra-color),
    0 3px 14px 2px var(--jp-shadow-ambient-color);
  --jp-elevation-z12: 0 7px 8px -4px var(--jp-shadow-umbra-color),
    0 12px 17px 2px var(--jp-shadow-penumbra-color),
    0 5px 22px 4px var(--jp-shadow-ambient-color);
  --jp-elevation-z16: 0 8px 10px -5px var(--jp-shadow-umbra-color),
    0 16px 24px 2px var(--jp-shadow-penumbra-color),
    0 6px 30px 5px var(--jp-shadow-ambient-color);
  --jp-elevation-z20: 0 10px 13px -6px var(--jp-shadow-umbra-color),
    0 20px 31px 3px var(--jp-shadow-penumbra-color),
    0 8px 38px 7px var(--jp-shadow-ambient-color);
  --jp-elevation-z24: 0 11px 15px -7px var(--jp-shadow-umbra-color),
    0 24px 38px 3px var(--jp-shadow-penumbra-color),
    0 9px 46px 8px var(--jp-shadow-ambient-color);

  /* Borders
   *
   * The following variables, specify the visual styling of borders in JupyterLab.
   */

  --jp-border-width: 1px;
  --jp-border-color0: var(--md-grey-400);
  --jp-border-color1: var(--md-grey-400);
  --jp-border-color2: var(--md-grey-300);
  --jp-border-color3: var(--md-grey-200);
  --jp-inverse-border-color: var(--md-grey-600);
  --jp-border-radius: 2px;

  /* UI Fonts
   *
   * The UI font CSS variables are used for the typography all of the JupyterLab
   * user interface elements that are not directly user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-ui-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: 0.83333em;
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: 1.2em;
  --jp-ui-font-size3: 1.44em;
  --jp-ui-font-family: system-ui, -apple-system, blinkmacsystemfont, 'Segoe UI',
    helvetica, arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji',
    'Segoe UI Symbol';

  /*
   * Use these font colors against the corresponding main layout colors.
   * In a light theme, these go from dark to light.
   */

  /* Defaults use Material Design specification */
  --jp-ui-font-color0: rgba(0, 0, 0, 1);
  --jp-ui-font-color1: rgba(0, 0, 0, 0.87);
  --jp-ui-font-color2: rgba(0, 0, 0, 0.54);
  --jp-ui-font-color3: rgba(0, 0, 0, 0.38);

  /*
   * Use these against the brand/accent/warn/error colors.
   * These will typically go from light to darker, in both a dark and light theme.
   */

  --jp-ui-inverse-font-color0: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color1: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color2: rgba(255, 255, 255, 0.7);
  --jp-ui-inverse-font-color3: rgba(255, 255, 255, 0.5);

  /* Content Fonts
   *
   * Content font variables are used for typography of user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-content-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-content-line-height: 1.6;
  --jp-content-font-scale-factor: 1.2;
  --jp-content-font-size0: 0.83333em;
  --jp-content-font-size1: 14px; /* Base font size */
  --jp-content-font-size2: 1.2em;
  --jp-content-font-size3: 1.44em;
  --jp-content-font-size4: 1.728em;
  --jp-content-font-size5: 2.0736em;

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-content-presentation-font-size1: 17px;
  --jp-content-heading-line-height: 1;
  --jp-content-heading-margin-top: 1.2em;
  --jp-content-heading-margin-bottom: 0.8em;
  --jp-content-heading-font-weight: 500;

  /* Defaults use Material Design specification */
  --jp-content-font-color0: rgba(0, 0, 0, 1);
  --jp-content-font-color1: rgba(0, 0, 0, 0.87);
  --jp-content-font-color2: rgba(0, 0, 0, 0.54);
  --jp-content-font-color3: rgba(0, 0, 0, 0.38);
  --jp-content-link-color: var(--md-blue-900);
  --jp-content-font-family: system-ui, -apple-system, blinkmacsystemfont,
    'Segoe UI', helvetica, arial, sans-serif, 'Apple Color Emoji',
    'Segoe UI Emoji', 'Segoe UI Symbol';

  /*
   * Code Fonts
   *
   * Code font variables are used for typography of code and other monospaces content.
   */

  --jp-code-font-size: 13px;
  --jp-code-line-height: 1.3077; /* 17px for 13px base */
  --jp-code-padding: 5px; /* 5px for 13px base, codemirror highlighting needs integer px value */
  --jp-code-font-family-default: menlo, consolas, 'DejaVu Sans Mono', monospace;
  --jp-code-font-family: var(--jp-code-font-family-default);

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-code-presentation-font-size: 16px;

  /* may need to tweak cursor width if you change font size */
  --jp-code-cursor-width0: 1.4px;
  --jp-code-cursor-width1: 2px;
  --jp-code-cursor-width2: 4px;

  /* Layout
   *
   * The following are the main layout colors use in JupyterLab. In a light
   * theme these would go from light to dark.
   */

  --jp-layout-color0: white;
  --jp-layout-color1: white;
  --jp-layout-color2: var(--md-grey-200);
  --jp-layout-color3: var(--md-grey-400);
  --jp-layout-color4: var(--md-grey-600);

  /* Inverse Layout
   *
   * The following are the inverse layout colors use in JupyterLab. In a light
   * theme these would go from dark to light.
   */

  --jp-inverse-layout-color0: #111;
  --jp-inverse-layout-color1: var(--md-grey-900);
  --jp-inverse-layout-color2: var(--md-grey-800);
  --jp-inverse-layout-color3: var(--md-grey-700);
  --jp-inverse-layout-color4: var(--md-grey-600);

  /* Brand/accent */

  --jp-brand-color0: var(--md-blue-900);
  --jp-brand-color1: var(--md-blue-700);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);
  --jp-brand-color4: var(--md-blue-50);
  --jp-accent-color0: var(--md-green-900);
  --jp-accent-color1: var(--md-green-700);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-900);
  --jp-warn-color1: var(--md-orange-700);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);
  --jp-error-color0: var(--md-red-900);
  --jp-error-color1: var(--md-red-700);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);
  --jp-success-color0: var(--md-green-900);
  --jp-success-color1: var(--md-green-700);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);
  --jp-info-color0: var(--md-cyan-900);
  --jp-info-color1: var(--md-cyan-700);
  --jp-info-color2: var(--md-cyan-300);
  --jp-info-color3: var(--md-cyan-100);

  /* Cell specific styles */

  --jp-cell-padding: 5px;
  --jp-cell-collapser-width: 8px;
  --jp-cell-collapser-min-height: 20px;
  --jp-cell-collapser-not-active-hover-opacity: 0.6;
  --jp-cell-editor-background: var(--md-grey-100);
  --jp-cell-editor-border-color: var(--md-grey-300);
  --jp-cell-editor-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-cell-editor-active-background: var(--jp-layout-color0);
  --jp-cell-editor-active-border-color: var(--jp-brand-color1);
  --jp-cell-prompt-width: 64px;
  --jp-cell-prompt-font-family: var(--jp-code-font-family-default);
  --jp-cell-prompt-letter-spacing: 0;
  --jp-cell-prompt-opacity: 1;
  --jp-cell-prompt-not-active-opacity: 0.5;
  --jp-cell-prompt-not-active-font-color: var(--md-grey-700);

  /* A custom blend of MD grey and blue 600
   * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
  --jp-cell-inprompt-font-color: #307fc1;

  /* A custom blend of MD grey and orange 600
   * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
  --jp-cell-outprompt-font-color: #bf5b3d;

  /* Notebook specific styles */

  --jp-notebook-padding: 10px;
  --jp-notebook-select-background: var(--jp-layout-color1);
  --jp-notebook-multiselected-color: var(--md-blue-50);

  /* The scroll padding is calculated to fill enough space at the bottom of the
  notebook to show one single-line cell (with appropriate padding) at the top
  when the notebook is scrolled all the way to the bottom. We also subtract one
  pixel so that no scrollbar appears if we have just one single-line cell in the
  notebook. This padding is to enable a 'scroll past end' feature in a notebook.
  */
  --jp-notebook-scroll-padding: calc(
    100% - var(--jp-code-font-size) * var(--jp-code-line-height) -
      var(--jp-code-padding) - var(--jp-cell-padding) - 1px
  );

  /* Rendermime styles */

  --jp-rendermime-error-background: #fdd;
  --jp-rendermime-table-row-background: var(--md-grey-100);
  --jp-rendermime-table-row-hover-background: var(--md-light-blue-50);

  /* Dialog specific styles */

  --jp-dialog-background: rgba(0, 0, 0, 0.25);

  /* Console specific styles */

  --jp-console-padding: 10px;

  /* Toolbar specific styles */

  --jp-toolbar-border-color: var(--jp-border-color1);
  --jp-toolbar-micro-height: 8px;
  --jp-toolbar-background: var(--jp-layout-color1);
  --jp-toolbar-box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.24);
  --jp-toolbar-header-margin: 4px 4px 0 4px;
  --jp-toolbar-active-background: var(--md-grey-300);

  /* Statusbar specific styles */

  --jp-statusbar-height: 24px;

  /* Input field styles */

  --jp-input-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-input-active-background: var(--jp-layout-color1);
  --jp-input-hover-background: var(--jp-layout-color1);
  --jp-input-background: var(--md-grey-100);
  --jp-input-border-color: var(--jp-inverse-border-color);
  --jp-input-active-border-color: var(--jp-brand-color1);
  --jp-input-active-box-shadow-color: rgba(19, 124, 189, 0.3);

  /* General editor styles */

  --jp-editor-selected-background: #d9d9d9;
  --jp-editor-selected-focused-background: #d7d4f0;
  --jp-editor-cursor-color: var(--jp-ui-font-color0);

  /* Code mirror specific styles */

  --jp-mirror-editor-keyword-color: #008000;
  --jp-mirror-editor-atom-color: #88f;
  --jp-mirror-editor-number-color: #080;
  --jp-mirror-editor-def-color: #00f;
  --jp-mirror-editor-variable-color: var(--md-grey-900);
  --jp-mirror-editor-variable-2-color: rgb(0, 54, 109);
  --jp-mirror-editor-variable-3-color: #085;
  --jp-mirror-editor-punctuation-color: #05a;
  --jp-mirror-editor-property-color: #05a;
  --jp-mirror-editor-operator-color: #a2f;
  --jp-mirror-editor-comment-color: #408080;
  --jp-mirror-editor-string-color: #ba2121;
  --jp-mirror-editor-string-2-color: #708;
  --jp-mirror-editor-meta-color: #a2f;
  --jp-mirror-editor-qualifier-color: #555;
  --jp-mirror-editor-builtin-color: #008000;
  --jp-mirror-editor-bracket-color: #997;
  --jp-mirror-editor-tag-color: #170;
  --jp-mirror-editor-attribute-color: #00c;
  --jp-mirror-editor-header-color: blue;
  --jp-mirror-editor-quote-color: #090;
  --jp-mirror-editor-link-color: #00c;
  --jp-mirror-editor-error-color: #f00;
  --jp-mirror-editor-hr-color: #999;

  /*
    RTC user specific colors.
    These colors are used for the cursor, username in the editor,
    and the icon of the user.
  */

  --jp-collaborator-color1: #ffad8e;
  --jp-collaborator-color2: #dac83d;
  --jp-collaborator-color3: #72dd76;
  --jp-collaborator-color4: #00e4d0;
  --jp-collaborator-color5: #45d4ff;
  --jp-collaborator-color6: #e2b1ff;
  --jp-collaborator-color7: #ff9de6;

  /* Vega extension styles */

  --jp-vega-background: white;

  /* Sidebar-related styles */

  --jp-sidebar-min-width: 250px;

  /* Search-related styles */

  --jp-search-toggle-off-opacity: 0.5;
  --jp-search-toggle-hover-opacity: 0.8;
  --jp-search-toggle-on-opacity: 1;
  --jp-search-selected-match-background-color: rgb(245, 200, 0);
  --jp-search-selected-match-color: black;
  --jp-search-unselected-match-background-color: var(
    --jp-inverse-layout-color0
  );
  --jp-search-unselected-match-color: var(--jp-ui-inverse-font-color0);

  /* Icon colors that work well with light or dark backgrounds */
  --jp-icon-contrast-color0: var(--md-purple-600);
  --jp-icon-contrast-color1: var(--md-green-600);
  --jp-icon-contrast-color2: var(--md-pink-600);
  --jp-icon-contrast-color3: var(--md-blue-600);

  /* Button colors */
  --jp-accept-color-normal: var(--md-blue-700);
  --jp-accept-color-hover: var(--md-blue-800);
  --jp-accept-color-active: var(--md-blue-900);
  --jp-warn-color-normal: var(--md-red-700);
  --jp-warn-color-hover: var(--md-red-800);
  --jp-warn-color-active: var(--md-red-900);
  --jp-reject-color-normal: var(--md-grey-600);
  --jp-reject-color-hover: var(--md-grey-700);
  --jp-reject-color-active: var(--md-grey-800);

  /* File or activity icons and switch semantic variables */
  --jp-jupyter-icon-color: #f37626;
  --jp-notebook-icon-color: #f37626;
  --jp-json-icon-color: var(--md-orange-700);
  --jp-console-icon-background-color: var(--md-blue-700);
  --jp-console-icon-color: white;
  --jp-terminal-icon-background-color: var(--md-grey-800);
  --jp-terminal-icon-color: var(--md-grey-200);
  --jp-text-editor-icon-color: var(--md-grey-700);
  --jp-inspector-icon-color: var(--md-grey-700);
  --jp-switch-color: var(--md-grey-400);
  --jp-switch-true-position-color: var(--md-orange-900);
}
</style>
<style type="text/css">
/* Force rendering true colors when outputing to pdf */
* {
  -webkit-print-color-adjust: exact;
}

/* Misc */
a.anchor-link {
  display: none;
}

/* Input area styling */
.jp-InputArea {
  overflow: hidden;
}

.jp-InputArea-editor {
  overflow: hidden;
}

.cm-editor.cm-s-jupyter .highlight pre {
/* weird, but --jp-code-padding defined to be 5px but 4px horizontal padding is hardcoded for pre.cm-line */
  padding: var(--jp-code-padding) 4px;
  margin: 0;

  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
  color: inherit;

}

.jp-OutputArea-output pre {
  line-height: inherit;
  font-family: inherit;
}

.jp-RenderedText pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
}

/* Hiding the collapser by default */
.jp-Collapser {
  display: none;
}

@page {
    margin: 0.5in; /* Margin for each printed piece of paper */
}

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }
}
</style>
<!-- Load mathjax -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe"> </script>
<!-- MathJax configuration -->
<script type="text/x-mathjax-config">
    init_mathjax = function() {
        if (window.MathJax) {
        // MathJax loaded
            MathJax.Hub.Config({
                TeX: {
                    equationNumbers: {
                    autoNumber: "AMS",
                    useLabelIds: true
                    }
                },
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                },
                displayAlign: 'center',
                CommonHTML: {
                    linebreaks: {
                    automatic: true
                    }
                }
            });

            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
    init_mathjax();
    </script>
<!-- End of mathjax configuration --><script type="module">
  document.addEventListener("DOMContentLoaded", async () => {
    const diagrams = document.querySelectorAll(".jp-Mermaid > pre.mermaid");
    // do not load mermaidjs if not needed
    if (!diagrams.length) {
      return;
    }
    const mermaid = (await import("https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.6.0/mermaid.esm.min.mjs")).default;
    const parser = new DOMParser();

    mermaid.initialize({
      maxTextSize: 100000,
      startOnLoad: false,
      fontFamily: window
        .getComputedStyle(document.body)
        .getPropertyValue("--jp-ui-font-family"),
      theme: document.querySelector("body[data-jp-theme-light='true']")
        ? "default"
        : "dark",
    });

    let _nextMermaidId = 0;

    function makeMermaidImage(svg) {
      const img = document.createElement("img");
      const doc = parser.parseFromString(svg, "image/svg+xml");
      const svgEl = doc.querySelector("svg");
      const { maxWidth } = svgEl?.style || {};
      const firstTitle = doc.querySelector("title");
      const firstDesc = doc.querySelector("desc");

      img.setAttribute("src", `data:image/svg+xml,${encodeURIComponent(svg)}`);
      if (maxWidth) {
        img.width = parseInt(maxWidth);
      }
      if (firstTitle) {
        img.setAttribute("alt", firstTitle.textContent);
      }
      if (firstDesc) {
        const caption = document.createElement("figcaption");
        caption.className = "sr-only";
        caption.textContent = firstDesc.textContent;
        return [img, caption];
      }
      return [img];
    }

    async function makeMermaidError(text) {
      let errorMessage = "";
      try {
        await mermaid.parse(text);
      } catch (err) {
        errorMessage = `${err}`;
      }

      const result = document.createElement("details");
      result.className = 'jp-RenderedMermaid-Details';
      const summary = document.createElement("summary");
      summary.className = 'jp-RenderedMermaid-Summary';
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.innerText = text;
      pre.appendChild(code);
      summary.appendChild(pre);
      result.appendChild(summary);

      const warning = document.createElement("pre");
      warning.innerText = errorMessage;
      result.appendChild(warning);
      return [result];
    }

    async function renderOneMarmaid(src) {
      const id = `jp-mermaid-${_nextMermaidId++}`;
      const parent = src.parentNode;
      let raw = src.textContent.trim();
      const el = document.createElement("div");
      el.style.visibility = "hidden";
      document.body.appendChild(el);
      let results = null;
      let output = null;
      try {
        const { svg } = await mermaid.render(id, raw, el);
        results = makeMermaidImage(svg);
        output = document.createElement("figure");
        results.map(output.appendChild, output);
      } catch (err) {
        parent.classList.add("jp-mod-warning");
        results = await makeMermaidError(raw);
        output = results[0];
      } finally {
        el.remove();
      }
      parent.classList.add("jp-RenderedMermaid");
      parent.appendChild(output);
    }

    void Promise.all([...diagrams].map(renderOneMarmaid));
  });
</script>
<style>
  .jp-Mermaid:not(.jp-RenderedMermaid) {
    display: none;
  }

  .jp-RenderedMermaid {
    overflow: auto;
    display: flex;
  }

  .jp-RenderedMermaid.jp-mod-warning {
    width: auto;
    padding: 0.5em;
    margin-top: 0.5em;
    border: var(--jp-border-width) solid var(--jp-warn-color2);
    border-radius: var(--jp-border-radius);
    color: var(--jp-ui-font-color1);
    font-size: var(--jp-ui-font-size1);
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .jp-RenderedMermaid figure {
    margin: 0;
    overflow: auto;
    max-width: 100%;
  }

  .jp-RenderedMermaid img {
    max-width: 100%;
  }

  .jp-RenderedMermaid-Details > pre {
    margin-top: 1em;
  }

  .jp-RenderedMermaid-Summary {
    color: var(--jp-warn-color2);
  }

  .jp-RenderedMermaid:not(.jp-mod-warning) pre {
    display: none;
  }

  .jp-RenderedMermaid-Summary > pre {
    display: inline-block;
    white-space: normal;
  }
</style>
<!-- End of mermaid configuration --></head>
<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">
<main>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=d8746cf0">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h1 id="Reference-Laboratory-Project---Section-11:-Classification-of-Blood-Cells-using--Convoluted-Neural-Networks">Reference Laboratory Project - Section 11: Classification of Blood Cells using  Convoluted Neural Networks<a class="anchor-link" href="#Reference-Laboratory-Project---Section-11:-Classification-of-Blood-Cells-using--Convoluted-Neural-Networks">¶</a></h1>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=b4b5dbae">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><strong>"</strong> <em>A tutorial on the extensive processing of labelled image files and subsequent usage with Convoluted Neural Networks to create a workable prediction model that identifies if human lymphocytes are deemed 'malignant' or 'normal'</em> <strong>"</strong></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=6160b4b8">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Ryan-Breen">Ryan Breen<a class="anchor-link" href="#Ryan-Breen">¶</a></h3><h3 id="October-5th,-2022">October 5th, 2022<a class="anchor-link" href="#October-5th,-2022">¶</a></h3>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=a13b941c">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="Part-One:-The-goal-of-this-specific-project-is-to-simply-evaluate-the-CNN-model-performance-when-classifying-lympocytes-solely-on-colorization.">Part One: The goal of this specific project is to simply evaluate the CNN model performance when classifying lympocytes solely on colorization.<a class="anchor-link" href="#Part-One:-The-goal-of-this-specific-project-is-to-simply-evaluate-the-CNN-model-performance-when-classifying-lympocytes-solely-on-colorization.">¶</a></h4><h4 id="Disclaimer:-No-real-patient-data-used,-only-public-repository-for-hematology-training.-This-report-is-solely-to-test-the-power-of-neural-network-learning-algorithms-and-in-no-serves-as-an-form-of-clinical-investigation.">Disclaimer: No real patient data used, only public repository for hematology training. This report is solely to test the power of neural network learning algorithms and in no serves as an form of clinical investigation.<a class="anchor-link" href="#Disclaimer:-No-real-patient-data-used,-only-public-repository-for-hematology-training.-This-report-is-solely-to-test-the-power-of-neural-network-learning-algorithms-and-in-no-serves-as-an-form-of-clinical-investigation.">¶</a></h4>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=890054c9">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h2 id="Background">Background<a class="anchor-link" href="#Background">¶</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=f5e348c1">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>Cell classification is the process of identifying specific blood cells in a patient's blood streams typically performed using light microscopy and a blood film ( a stained artifact of a patient's blood). This project is the initial proposal and modelling of such white blood cells to determine the malignancy of a specific blood cell called a lymphocyte. Malignant lymphocytes are virtually associated with cancerous states in patients where normal lymphocytes typically constitute anywhere from 20 - 80% of the white blood cell population in human beings. <br/><br/>
Furthermore, blood test from a commonly ordered laboratory panel (a set of tests) called a complete blood count has been integrated in the model to see if the presence of such values create a more accurate or predictable model. Note, that a spectrum of abnormality in terms of cellular appearance exists for white blood cells where cells noted as 'reactive' are non-malignant, but appear as an intermediary form between malignant and normal cells. Reactive cells are typically cells just 'doing their job' anthromorphologically speaking - these cells typically respond ot viruses in the body. 
<br/><br/>
Below the the difference between 'malignant', 'reactive', and 'normal' lymphocytes are depicted. Note, a general trend exists where malignant lymphocytes are more <em>immature</em> cells that are precursors formed in the bone marrow that <em>pour</em> out during malignancy due to the increase in the number of malignant cells in the body. The malignant cells, as well as immature cells, are typically larger, deeper in the blue/violet spectrum, and have a more coarse appearance as described by a more pourous nucleus (the darker central button containing the DNA). Normal cells are smaller, lighter in the blue spectrum, but have a marker finer darker nuclues. <br/><br/>
For the purposes of this study only appearance in terms of color intesity and and color hue will be used as distinguising features as opposed to using cell size since cell diameters have not been provided. The colorization of cells is detecting using numerical bits representing a color hue in the Red-Green-Blue(RGB) schematic commonly used in most computer systems to produce colors using 8 bits of information for each red,green, and blue combination. Overall, this produces 2^8 or 256 possibilities for each of the red,green, or blue color choices and using independent assortment a total of 256 x 256 x 256 = 16777216 total possibilities. Note, that each bit within each color acts as a sole estimator within neural network where the neural network creates weights using the <em>Convoluted Neural Network(CNN)</em>. <br/><br/>
The CNN modelling algorithm utilizes.......</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=ebaba015">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="Normal-Lymphocyte-(note-appearance-may-be-larger)">Normal Lymphocyte (note appearance may be larger)<a class="anchor-link" href="#Normal-Lymphocyte-(note-appearance-may-be-larger)">¶</a></h4><p>The lymphocyte is the 'purplish-blue' cell with the darker nucleus region of the cell in the center and what is referred to as the cytoplasm on the outer range. Normal lymphocytes may have much larger light blue cytoplasm that are commonly referred to as 'large lymphocytes' that are completely 'normal' pathogenically speaking.</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=f7b6fed4">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><img alt="N%20%284%29.png" src="data:image/png;base64,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"/></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=6cd5a7e2">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="Reactive-Lymphocyte-(appears-in-infections)">Reactive Lymphocyte (appears in infections)<a class="anchor-link" href="#Reactive-Lymphocyte-(appears-in-infections)">¶</a></h4><p>This particular cell type serves to test the robustness of the model by exhibiting characterists of both normal and malignant cell types. This is the initial phase of modelling using CNN to classify cell types. It is expected that subsequent modelling and possible data exploration could be necessary to explore effects of such intermediary cells. Again, red blood cells found in the background of the picture, normally found in blood films, may also create diminishing power of the model, but may be subsequently investigated in the "post-hoc" investigation by performing image processing such as diminishing values in the "red channel" to disipate the red cells from the background. However, staying a firm believer in rducing the number of variables in a study, the initial model will simply contain cell images "in-situ".</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=8df160ee">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><img alt="N%20%281%29.png" src="data:image/png;base64,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"/></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=5df0c181">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="Malignant-Lymphocyte-(Blasts)-(present-in-cancers)">Malignant Lymphocyte (Blasts) (present in cancers)<a class="anchor-link" href="#Malignant-Lymphocyte-(Blasts)-(present-in-cancers)">¶</a></h4><p>Malignant lymphocytes are either morphologically altered normal lymphocytes that have changed due to a pathogenic environment (presence of molecules that prompt cellular change in the face of malignancy) or immature lymphocytes that are precursor cells to the normal lymphocyte. More common, the precursor cells depcited by a chronological lineage of cells, contains morphologically/visual changes that somewhat organize as one examines the cells further back within the lineage. As cells become more immature, the colorization of the cell tends to be darker in nature (dark purple and blue hues), larger (to be measured in future studies), and contains more abnormal features within the cells themselves. Most commonly, the features within immature cells is a very large nucleus that is described as "coarge" in appearance - almost with a grind glass cartoonish like sharpness. Furthermore, as seen with the cell below, abnormal additive features such as the bubble like inclusions seen below called vacuoles are common in malignant cells.</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=3147cbbc">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><img alt="P%20%2810%29.png" src="data:image/png;base64,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"/></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=957438dd">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h2 id="Purpose">Purpose<a class="anchor-link" href="#Purpose">¶</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=66b8a4e5">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>Classifying malignant cancer cells is typically performed by screen such cells under a microscope with a special staining of a blood sample from the patient by laboratory professionals. The process works well when performed by trained experienced individuals, but like all human endevaors is open to the possibility of human error. Another constraint on medical laboratory screening of cancer cells is the cost factor alongside the laborious process to create such slides. Reducing the blood film to a simple .png image eliminates materials and time needed to created such slides. Furthermore, digital blood films are able to last indefinitely and may even be sent electronic much more rapidly for further consultation.</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=fb274d39">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h2 id="Obtaining-and-Processing-Image-Files">Obtaining and Processing Image Files<a class="anchor-link" href="#Obtaining-and-Processing-Image-Files">¶</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=33110d1c">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="Major-Goal-Convert-.png-imags-to-NumPy-arrays-with-shape-(width,height,channel)">Major Goal Convert .png imags to NumPy arrays with shape (width,height,channel)<a class="anchor-link" href="#Major-Goal-Convert-.png-imags-to-NumPy-arrays-with-shape-(width,height,channel)">¶</a></h4><p>Below, highlights how each NumPy array should be structured for use with Keras modelling. For channel last conventional arrays, the image attributes are listed in the following order:</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=8c2d8c5b">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="As-an-example,-a-28-x-28-RBG-format-png-is-depicted-as-an-array-below:">As an example, a 28 x 28 RBG format png is depicted as an array below:<a class="anchor-link" href="#As-an-example,-a-28-x-28-RBG-format-png-is-depicted-as-an-array-below:">¶</a></h3><ul>
<li>image_array = _(num of samples, height(pixels), width(pixels), color channel) _</li>
<li>num_of_samples: <em>The number of image files within the array</em></li>
<li>height:<em>The height of image in pixels that is 28 in <code>NumPy</code></em></li>
<li>width:<em>The width of image in pixels that is 28 in <code>NumPy</code></em></li>
<li>channel: <em>The method or schematic to represent color in this case the RGB system explained earlier</em></li>
</ul>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=b58371bf">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><img alt="NumPy_RGB_Array.png" src="data:image/png;base64,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"/></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=37fb477f">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Libraries-and-Precursors">Libraries and Precursors<a class="anchor-link" href="#Libraries-and-Precursors">¶</a></h3>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=e08d5344">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1">#conda install OpenCV</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=c264c9d4">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [110]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1">### DELETE ME AFTER ###</span>
<span class="kn">import</span> <span class="nn">os</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">from</span> <span class="nn">PIL</span> <span class="kn">import</span> <span class="n">Image</span>
<span class="c1">#import cv2</span>
<span class="kn">from</span> <span class="nn">matplotlib</span> <span class="kn">import</span> <span class="n">pyplot</span> <span class="k">as</span> <span class="n">plt</span>
<span class="kn">from</span> <span class="nn">keras.preprocessing.image</span> <span class="kn">import</span> <span class="n">load_img</span>
<span class="kn">from</span> <span class="nn">keras.preprocessing.image</span> <span class="kn">import</span> <span class="n">save_img</span>
<span class="kn">from</span> <span class="nn">keras.preprocessing.image</span> <span class="kn">import</span> <span class="n">img_to_array</span>
<span class="n">bold</span> <span class="o">=</span> <span class="s1">'</span><span class="se">\033</span><span class="s1">[1m'</span>
<span class="n">end</span> <span class="o">=</span> <span class="s1">'</span><span class="se">\033</span><span class="s1">[0m'</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=55f980c2">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Create-File-Directory">Create File Directory<a class="anchor-link" href="#Create-File-Directory">¶</a></h3>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=90c630de">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [70]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">os</span>
<span class="n">neg_file</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">getcwd</span><span class="p">()</span> <span class="o">+</span> <span class="s1">'</span><span class="se">\\</span><span class="s1">neg</span><span class="se">\\</span><span class="s1">'</span> <span class="c1"># directory forneg images</span>
<span class="n">pos_file</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">getcwd</span><span class="p">()</span> <span class="o">+</span> <span class="s1">'</span><span class="se">\\</span><span class="s1">pos</span><span class="se">\\</span><span class="s1">'</span> <span class="c1"># directory for pos images</span>
<span class="c1"># imagine we only want to load PNG files (or JPEG or whatever...)</span>
<span class="n">EXTENSION</span> <span class="o">=</span> <span class="s1">'.png'</span>
<span class="n">neg_files</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">listdir</span><span class="p">(</span><span class="n">neg_file</span><span class="p">)</span> 
<span class="n">neg_paths</span> <span class="o">=</span> <span class="p">[</span><span class="n">neg_file</span><span class="o">+</span><span class="n">file</span> <span class="k">for</span> <span class="n">file</span> <span class="ow">in</span> <span class="n">neg_files</span><span class="p">]</span>
<span class="n">pos_files</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">listdir</span><span class="p">(</span><span class="n">pos_file</span><span class="p">)</span> 
<span class="n">pos_paths</span> <span class="o">=</span> <span class="p">[</span><span class="n">pos_file</span><span class="o">+</span><span class="n">file</span> <span class="k">for</span> <span class="n">file</span> <span class="ow">in</span> <span class="n">pos_files</span><span class="p">]</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=6fba5d71">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Create-the-Arrays-using-Pillow">Create the Arrays using Pillow<a class="anchor-link" href="#Create-the-Arrays-using-Pillow">¶</a></h3><p>Divide the 255 values to standardize by max value 255</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=9e774b0d">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [71]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">neg_images</span> <span class="o">=</span> <span class="p">[</span><span class="n">Image</span><span class="o">.</span><span class="n">open</span><span class="p">(</span><span class="n">path</span><span class="p">)</span><span class="o">.</span><span class="n">convert</span><span class="p">(</span><span class="s1">'RGB'</span><span class="p">)</span> <span class="k">for</span> <span class="n">path</span> <span class="ow">in</span> <span class="n">neg_paths</span><span class="p">]</span>
<span class="n">neg_arrays</span> <span class="o">=</span> <span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">image</span><span class="p">)</span><span class="o">/</span><span class="mi">255</span> <span class="k">for</span> <span class="n">image</span> <span class="ow">in</span> <span class="n">neg_images</span><span class="p">]</span> 
<span class="n">pos_images</span> <span class="o">=</span> <span class="p">[</span><span class="n">Image</span><span class="o">.</span><span class="n">open</span><span class="p">(</span><span class="n">path</span><span class="p">)</span><span class="o">.</span><span class="n">convert</span><span class="p">(</span><span class="s1">'RGB'</span><span class="p">)</span> <span class="k">for</span> <span class="n">path</span> <span class="ow">in</span> <span class="n">pos_paths</span><span class="p">]</span>
<span class="n">pos_arrays</span> <span class="o">=</span> <span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">image</span><span class="p">)</span><span class="o">/</span><span class="mi">255</span> <span class="k">for</span> <span class="n">image</span> <span class="ow">in</span> <span class="n">pos_images</span><span class="p">]</span> 
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=3e696708">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Explore-Data-Shape">Explore Data Shape<a class="anchor-link" href="#Explore-Data-Shape">¶</a></h3>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=786d93d9">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [72]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">neg_shapes</span> <span class="o">=</span> <span class="p">[</span><span class="n">array</span><span class="o">.</span><span class="n">shape</span> <span class="k">for</span> <span class="n">array</span> <span class="ow">in</span> <span class="n">neg_arrays</span><span class="p">[:</span><span class="mi">5</span><span class="p">]]</span>
<span class="n">pos_shapes</span> <span class="o">=</span> <span class="p">[</span><span class="n">array</span><span class="o">.</span><span class="n">shape</span> <span class="k">for</span> <span class="n">array</span> <span class="ow">in</span> <span class="n">pos_arrays</span><span class="p">[:</span><span class="mi">5</span><span class="p">]]</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">'First 5 shapes for neg images:'</span><span class="p">,</span> <span class="n">neg_shapes</span><span class="p">[:</span><span class="mi">5</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">'First 5 shapes for pos images:'</span><span class="p">,</span> <span class="n">pos_shapes</span><span class="p">[:</span><span class="mi">5</span><span class="p">])</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>First 5 shapes for neg images: [(177, 177, 3), (160, 224, 3), (115, 113, 3), (115, 113, 3), (115, 113, 3)]
First 5 shapes for pos images: [(86, 109, 3), (92, 108, 3), (99, 91, 3), (99, 91, 3), (99, 91, 3)]
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=f69ff698">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>As we can see we have imags of different size widths and heights - this will be a problem for the Keras that requires a square input. Let's crop the images before exploring some more.</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=f61ed217">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Resizing-the-Images-with-cv2-Library">Resizing the Images with cv2 Library<a class="anchor-link" href="#Resizing-the-Images-with-cv2-Library">¶</a></h3><p>Note: ther image processing will be tried post-hoc after initial modelling</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=a7b01f93">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [73]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">first_neg</span> <span class="o">=</span> <span class="n">neg_images</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
<span class="nb">print</span><span class="p">(</span><span class="n">bold</span><span class="o">+</span><span class="s2">"Original Image"</span><span class="o">+</span><span class="n">end</span><span class="p">)</span>
<span class="n">display</span><span class="p">(</span><span class="n">first_neg</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre><span class="ansi-bold">Original Image</span>
</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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
"/>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=33450b3a">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [74]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="p">(</span><span class="n">width</span><span class="p">,</span> <span class="n">height</span><span class="p">)</span> <span class="o">=</span> <span class="p">(</span><span class="mi">100</span><span class="p">,</span><span class="mi">100</span><span class="p">)</span>
<span class="n">neg_cropped</span> <span class="o">=</span> <span class="p">[</span><span class="n">image</span><span class="o">.</span><span class="n">resize</span><span class="p">((</span><span class="n">width</span><span class="p">,</span> <span class="n">height</span><span class="p">))</span> <span class="k">for</span> <span class="n">image</span> <span class="ow">in</span> <span class="n">neg_images</span><span class="p">]</span>
<span class="n">neg_arrays</span> <span class="o">=</span> <span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">cropped</span><span class="p">)</span> <span class="k">for</span> <span class="n">cropped</span> <span class="ow">in</span> <span class="n">neg_cropped</span><span class="p">]</span> 
<span class="n">pos_cropped</span> <span class="o">=</span> <span class="p">[</span><span class="n">image</span><span class="o">.</span><span class="n">resize</span><span class="p">((</span><span class="n">width</span><span class="p">,</span> <span class="n">height</span><span class="p">))</span> <span class="k">for</span> <span class="n">image</span> <span class="ow">in</span> <span class="n">pos_images</span><span class="p">]</span>
<span class="n">pos_arrays</span> <span class="o">=</span> <span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">cropped</span><span class="p">)</span> <span class="k">for</span> <span class="n">cropped</span> <span class="ow">in</span> <span class="n">pos_cropped</span><span class="p">]</span>
<span class="n">display</span><span class="p">(</span><span class="n">neg_cropped</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="n">neg_arrays</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
<span class="n">display</span><span class="p">(</span><span class="n">pos_cropped</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="n">pos_arrays</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAABBlUlEQVR4nF39SZcuSZIdiF0RUTWzb/Lp+RtizIwsJDIrCaCJwiH7ACQXXJFc8q/yHC560U2wSDT7EA1WAeiasivHyIzhDT59k5mpilwu1PxFNjwW4c+Hz+1TUxW5InLvNfm//Xf/nx+vbCgTgxSYKKqHSYAKeoRABBARQEWhFuEqTJdfDJeXA6oQpJCEBEAQCCdBEqIiIgBAgIRg+RCBgMHlu4L2PZJKEBRTUMAgoYKIAFGdiCgTypN7LexUjpVqaqpKP4MZebA8mLtLJ3mlXZdERaS96vMfb/9j+4wABEIAEBUhQLJdY/tyDfmfH+33e0mfq2+ncylzLRHhLkIgTBKFjIhgiJpCCEEyqZQoQjEeZ8dczxin0cmcskDzSvu1aWfCiFnoCRICiCgYIIMQaZeiqkKCDEBoEBEKJECSTlMjAyICBqNWV6eTfvC6d6JoNTni/FgYyJea+oTE4/va1c62li6VmZDOchKTtkZkez1+XLXn2ydBikAgbanYbiIBICv+/ELPdUi6vxuFpdao0dY+FBBpLwuRqEE1y0JxLyiVXtB16fGPhw9nGU9evBCSRAjkTi5fbC8/WW8us0AIZw0AEFuuigGFAATcoaJtbYQKad9RaRvU4+O7gpPVhVJiLo9lupsCxFxN1WdMU/jA69cXtjGvcv5uEi+7i42qepnFxXqhhlIjaCoCgiAZClEVAYISAaCdLkhbBCBoogQy6z+/9HQeS5WZLkmX9zOOY98PREBFRdWgEoSVIpxLqQEwDTwfMR1YPWgVREAYOp3lcL9/+P5w88nu6tP1sCWpDIugCESeNwoAIIJiIgJChESQBFI7tqSADBFpdxptBYsHIw+pFtZS5zLptt++WKc1IERUIlRcu5A5ZMxuRUWUCQBZgYgQFWuvRmcAAIUUst1CaoBQBwUEoy1acIVI7jSDQZ/DCYbcK0EIHeEhAk2oxeexRAkyUrY6F6VCqmXRlMJZERA6CwMP93jazy+fptc/2a1vVhK0qs5oASz4MSgIGSRE22YLgPRowUS0rRkFLWaylIJQTVq1RswpWS0WAmyqRJq+r+gRo2qnKXfTvop2eZfcaUFRkedbEIRARURMakT7AwaQwPMZFS5nENpuFINM4qpiqh9jL1R++Hw5zyFlLLUU0ERE1AyhJv0mi6o7pdMuOM+zMxzhAtR4+3Y/k6/Ur2/WxuyOJS6IRFAVahoRDJq04C8UURFnoEWWtgUFJFUhQjGmQf0kUanJ8rXlm6Q7lXDvsoNyqsVnPab+skvDkjXwvOIqCgABIkRNiBQRArW216IFIAioLX61JQQBqCTVCITB8F+skErXGSS8sM4RQYZ6RDdI6tjlnLrsQgTqXC3R1OSopQi83Z2opdx9d0CJ/n+VV9eqNEaIKCNIRgBwEWlLFkFVEdWPKUvAFjTbMVTVnDO06krqJgyWJSGLriEVrGKUcqp1qnmTMUCHZF3LF8/ZL1wEIiraXrOdwI9x8zld/hdbhRRVMa2lpJREQz5mB/7wQ1C1QHh4qaiFEWIqq6Fbb1Lfd9Z1Y53FmUzUSqDqGGQ0iBARKlrLfPc98ip9/ueXq9S78PkvsCUgE23n0WwJVRCRBiw+/iAEgKhaOzWZ+SKJDX5wAaVieph81JwYY1XVbmvYJO+kspioiLT1UlW2uyQqqi2cN3gT0Xa2/OliPadMIckaIFKSgD4HLAVi+dQMopPPMU8shYSkjDRYt87dRizRzDuhA2oJJj7PXotXDXcgVAUINYPk++9Pq529+TGyrIp0IZOYETAVcLnHERQVU/UIChQgGU4Aqg1U0LKmUHplB4t02s+lBlymfZUQM6uTpC1tEB3EsiZVaKiKKOiVaqS0m0EuEE/QsEt788saKdHQQ/uJ5V4J0p/sJSy/1a7epYxSJ3gJgiLMva52w3rbpxxmySsFaRgIxDz7fJrPY/VKCMlQTREUCaRpmvHhu/nilV/tlBF8vi53KkJVRVXoIN2dYMM37aACdA9VRHjbIWpqjDpYerXBNHGSxITstsZm6NBTVlmTAO4VIpQuKFRREgHK84blx63LJYT/sKG4nNAfPlMwmFpuWDYfIQrLBjL32g9Wi/Bwjqoi6FfSJZhCkQQ5JafMKUUZfXqsp7syjREehEMEhIqSRDhET4c4P8bmqkpNwgbLASzIQMgW9YEfUL6IYkGG5HOiVBWjOYOMtIatOh1VgHqCdmJbqKUliUCIj2e6xcDlmP+X8Wb54vJVPv8Qnv/4RzibXv74anyax+PUXm511W2u+qSp780ywv3tP8rd/QmdSmJKAlHLZqn2KynB/bvp/vvz4c7HU61aVVSTtksLAiIMQoJeDk/zxexJBU5GhSxZl6RHqAgZqkZEOyBsJROh2sINSJq1GAQrXlldiR751la1C9B6EqCTDIqaacqaUlsviXBpgfzj8i2FTgtdERSYaQQQVIgALtoiAgzh6eL28sXLCHeCoiKmBNXMaGJVakhBmTxng5qaWQpLvlpLmfzum+n73x/3T5O3uKwCaWBFRAUeAD2gqiBr8XBKZstyAMOpKi3u4hl2tXKyYccFNSzFEpf7DKqRqloSoooCtLSCB0wVJE0gpiYpW0oNsAMMMJZap22dJf9RTExsgUxKCiBWSQFFKUQwABVIqqVoNlWTdimVIhrFKZJW6s55ngF4BCC5i65XTXx6e/72d/v773ysKCKmVdQTEmCyrNbHONo2s7O6kAqJBRaLKoJALFtJtd3cH6Jv++QZxC9xpiFbg1jXeVj1qghvQYep7WZLkrssShE2sBKxpHx5PmFkA25iycTEeQ56nYoEretUzSyJGgG4RCWAZKoKjQgiTJQURGsHRIT4VKepAqICEwRKnfjwbX34/dPDYxkZlpHAVocCBBb8zXg+RwAoKpYkWwDB0B+iaQtKAiWjupsaRJwuzwlHFc+NCyH4EXS1YyBiqhopAJFo1YCoSMoGARCAhFeIqEo4n0+egAIRqueOdcbx/nT8cDw+ztMTA8i9bi71+pPLzdVKVMQoUIakYHiQZEv2xDOWNgXgs88FBJK5Jo0aj+/P++94N87sPAEeroSKQkBEBIXCCFWDPP+nNfXWrV0zRCB0QJTKqKoSQHUXhAqCHhEhXIKGKLD0ZwRAkIEwwESCEbFA69bySapLJYfwKoIGPsNDVfH8TzIkqSg0KYTHx8P739fvf/s4HucaEUEPSoiqXPz29PKL7e2PV+vrQUh4SiCWsEE6qQo21GYmQpApqSNS0mQ2H+eHp2ki1OCQ8KoAuKAk9zDTdqIjXNTUoMZhLd2Wc6r7Y6w2c0Q1TejNJySSpApFrb3zCBoaHv6YkJ7LNhE3CKBOoL35FqrbsW21MJacho9IafmSVxcRVWWopKJWp7v6m7/aP7x/Yk3VQHVNjiAo7vbh/Xh4jPt3x69+fnX1+dY7TS1xtcPQIhlJXRC0iEogVCUlYdTpGOOsVVyN4RWogLW3Y6Ite7m7ANZOO6LrcyR9OvDtP/o/HO4uPtOLK93eDt1u2m5tyFmTmIA1li6XtU4g20kBlrOnIgoUhLUWRTtRIiQboBPRBXm0s7F0q5Z1BNrRQTBE1Ezrvv7+H95///2TqGsXIozKgFCMDMkkzlP1b7/u6A8/3fbbqy61iNBiyw/AH6ilJFXJue/THLNZlMnPxwKCkHBhTXA4g+FqWsJTMlWJqISSkjvvuno8xf0f9O4d3j2cctKX8+5q12++3XdJdpfd5rbsbmx33XdJoRDSAg3XOkUhKlUgEUpDtOjRuoULflqwVDsY7WtLt7N9N1oQkwiodB4VYMpSavnwx/27P56YaClBIkoroAWMYMBDDYFgrW/fxc2vHzb/StIzLvsIYp83cEgwoJZ6m090F591nvx8HoMSDq9OLg1lBgiGewTVxJTW7cPz1/9of/PX999/azrq7opf/VeXuxcTV3UmUNd3b+fx7E8f0K/n3dV6e5Fy512nllKoACGs4a3XHUFEhEFUEERUF1lOqrZuXitKiVggAPHc6lzaZwwITaFWT3flD797IM1yJRE1SEADEUv+IEAFQqx4Td/94XT95TpFS9dLv5si+nwAVeCaNGeDRymYz6xuERJkkM8L/F8CboCWOB77v/uPx7/+q+ntgyQ93ZpsP/THXx63E7Zf5vVna73APHMOxhHjoRz2x80mdz36laU8ddt+PQgMYoYkglCXVm2IGiL4scBuJ+JjkfQRF+B/0UFul2wGyzafp/e/ORzvqvau0l4qRCEiLcGxtSElVBVgjfr4yA+/m1KrcUhQyRCQ2op+CQTULGcT0agYp3kcKyCMWLYVIJClYveqGKyvauXhXfyH/1B+//cV0/iTW6hrR98NkFn3v6zzu4H33P4sdzcZTGYZFSL1PKGGzmdxjsO6jqtOxfMGaaXDkLKJJQCooCgk0KI1GC2G0QUa+kN8J4JLj2Bp6AkUtHh4f/7w/TElCxtBMxOIeEQ7zxoLAAyS8LYN4PHwdkwg2giFAVNhMCIIhkBFKZE3gMHnFv7ZgoWl5OESJGSBxIChZqvff2d/+W/LL39/3PV+3ftFEhNZ97oe1k7Mlf4wjX8dF2/z7Z9vd5/ZcFtLJ5gNdDpCnUqvPJ8c4dyHCPt16tcpd9IN2TK7TigMloYv4EBoa+CHNxzWKs3ndv/S+BHRmE7z29/fn6eJBkMiowFEFZUGlBEgJMIEEGWIaSpenp7SEuCj1GSJIAEPN5pkjaCQw0XuBp2eXEQoQv+hbG93LIkomJLmdfnmO/x3/835d28ft50MKReVSbyrenHbrS71dEDZ09bhzg+/Hac73v6T+cVXKb3oYt1Zb0iMQq81TCxpUOpsXlhKnPczhJrKapf6DO2l681SzX2nYjSqECHaWhMfmwpscKJFmtAUp+/mp3czxJNJLGBNglQVEUmmoYqgqoEMBkVBsQRnSUEioKpeK1RFkCyRS9AP0Ib+4nL7sN8bDO2Ety6mOBWMUFFFdEmfHsf/4d+dfv+tXg78+ZWYHg8unaX1phs2OqyRO51SCSCS09J0rIffFN0Ps527zbD5Ude/SXmVc86wOgWFAmUe0lxLbYOf0ae5qqZkKeloNm+vqBvpeulzUoel7B4BtpzYulUNaaQUEfPx/Rijaa+BSSTJUpnBPVrJhVaGixDRNgXgZGhYEpGgqyhFdAmIDBIFpupCS7J51Y0HPd6pmcwzyFAxFRUVpFCHqXja//X/b/qHX2HI4y57yXW33qyKsJOc0Q3JDGTkdacSqxfk6DzlufLD4xxHRBH+9dP6Tb7+qhtuN+lGVy8kDRlmEbVLrEVB1eQh2sLmeaoecTxPMBtWXG8jZaRUV0OnWSSJQUBqG+mIIus0l8f9yDQFgEgEI5ZxSEq25ANKCCM85RTBJa1CMCOB/Ngpfd65EEggnAGGzpG7fvtqMzuDmKcqrVGFLFIhXeqOZvH3f5f+01+L9Id/fhNSV4fx/Jhkm2Qrqc64fxe7q5R6dUedKkPzRl3HsK7K3O3ENlYfZbyrd0/VUXRtF592q5e5u5Dhldku9QbJrl3SgnlfKpByZ5bKVMTLfq+nd14DpnM/lLzJac3dYGkj3Vr6nMAiovNjOR9KFZoqiVpd7blrBqhoMKANXi/Ity2JWRKRFGRaatXnLEJCxGAEVU2omrC5HSQCfhwP3TQBOkHCTNMwd1369d/wv/+3x6cz31yOQ06WuNENBfPkJ6dqrqHx6BeXRkZ0tv8g+uTSUzS61CFKxSG97FLvMfe1Vqvj4Zvp3V+X1HXdaxuubPdZd/Gjbf8ics7sZiBHchUYwFlAlBiD6kX3h7G8PUFjZblbd7s3+dUntr0AOO/fPk5Hl9YUopiZ6HPn/fkjggRbBBdIC9EpJZi1QTLiuRJbGmXA0kKiBBE1LNn2ZhOQUvT43t1oa2fWw1j/9n8s/6//dv/h8PDmUl6m1ft3abPj9VX0qT+eopa5y2adTMXPhymvTZxyKd0242jTdMLGeOqm+7nsNAlrL6tLX91a0jTO01R8Ple/t/t/POdhXn0S1z/u9Dp1m6Rm3SoNG50UJHJlrdXDAh5nKyd48un+9Pt/rO/edD/9319sV/Pj+6lEFXPWaLmvddaWsokU4DmT8mOrXkVI1R7peSrF59LhedoPqra5XpXWKze7uN11q/54nCzf9NvTr/+2/NV/E//vf/cwTfpq2P3iFdZDvL/rbPA0eNRCKNBFCKOQEVG3u005Ci36HMx1tdbuRv0J8n2ajrWK9Z9OF3kQ1fO5Riqb666/UvMx3uLxfjz8bXr81ayqQ+7Qm9/EzY/y6jrbrcnak3fh1CIS9OqBqHN8/6vpb//y8Hjkz/8l5gliES3mhwhDnzsmJBDxpyXnMjcEBebu3UqTiJgtw9/nNpu0GK8NVRkAMLQ6LXN9sep3g3ZRGHdfz4dv482QrZ5fDLlXHXL55JOZmlW6Uuquj6fgHMyVmlPX9wwc5/Lu7byyvut9tckD2W3ZfVl7dOfK7iL7UzzdTacphHCpp5PoQarI6mao47kEs5rvok7y9B/j8T+ktJkuvtisv+yH154ucqSiXbcSHc/1PI92aYdv5W//6uH2TbfqICd3F4UQbKH6T+eEH6cYPzQvIQAsYrvWxGCL+CIff3QpU5fySmUBrgKvCodqMpT7J50f9PYX8uLL/vxXsvlJHdb54VeCWtFXdtpbur7G6kJPR5wPcxl5DCnupz0+3CeAV1cxHKq9582bYRg8nKdaj3fzUGUyhQYE9+8ny3ppBq3dgFUIk0jvw7Xn3KOL84e5nOL9757km2677oZbXV1lv5nXr/N6O8wpx2V8ubbzu/PTvq62lpLVaGgx2JrMDbsKIGKqQdJ9Kch1OYmp02GTEyHeQv+yxMtCe7iIqGg4RNgSfwQrXREQTMf5PE3dsLp47fuHlT3d5w3m+Xw+5UTzyc85us76wTaX43mK+/fB4PW1XO5iN6bHfbU0hybD8PQhHj0VnEq4hFxtcreJ9RWL2vkpepVpG/OI8CKdQaSeOM2S7Ly+WvdrzvsYHX6Mh2nff4/7b+Xw0PW3cvvVAdvobpJNenjoDgd5uZOUxWrM7gLQG10ImsTMlkiEZUjgCEiDnkKKC5K0fu0P9TCfS3l93pBtUiDhbTwLEhE+jiiVLF5v9OKnafr/bu7+7jwM2+1n42pYl7Mc9/7+HX2cdi/m1JlucHisebKXr+LzVxPPaR5tfaGQOs8sbn2WDjKHu/blVM8AEzT7VDieNKOPsY7HUTWDcp5Ai009pg42xCqLXeJU67jPrEXt9PSA+/8+TTPSKxs+Tffj5pu3+fULdIY/4XXwY0HeyGOMwPPcKTzEtPX5tMZ8qonPfeuFQ4KlFaQqz6DWSEYERFrvQrXx9rTLQ9efjJvVreNfGf4SZc/dm/7iutOt9328f+enfU1TebH1LyDf1u5+9N9/k778JD77kb69l/Ox9r3kbLsdWQ1zZGXYHIL372nGFy96oJQ9SlTpRY2WWQpGeBZ5fO+54/ZGjVZr9Br7Ry9j6W5QBz2ep5PX8ke1t3ne5ref6zTrqlumX88DRCwonC5trB/8GMGxHFYFFJXpT6LbM2GqjaYZhKMN0sGFfuJUk1Z8mqlYqGKTtB9s+OmFjvbdvz0cf4url/nidbfe8PL1+f6b9O0f5P7OL3e42dhxxtf3s1e8vNEXt9JXHREp6XoV7x9mt9znLmKUrguRqD5j3rzImwt9Orsxatf3ovbgYpKlVLcp9Okh/Dz7PL/+or+8tfNTNwzydDc9nU7o6sv1Zh7rr+6O26PSk4e5u4c/D7QaMyx0oQxAfgjzS5/RSSZBF+l5cVsJygCW7BdsJzGWm0AEW7f5Gdd66n2l1vdqxpT7V/8i+4f0u//xw4c/nNMwdCLbnU1rZan3x5gTOveLvhzD/nBf3p/qxba+HHrt43LFDh3meDrG0KV+SNmypiijPz357OxzlTQw3O95Ulc4hRUaCtGY5zq6FnZ4ss8/Hdc3lhjHCd899rOkuSshOp90fMgRuzaafO6APfee2QiHP2TGZYgJqFgtRCIpKaLN7URab0KXhkJjwiwJtJG7WmcHiCC9WuLVzSqLZ4Mok/Zycbr91zI9rt//5mBRbq7WHPT+27PJfH0ZQ5+yhllUcm9ZUpWQb7+vZ5Hxxoa+VuqQ5jqW1WAmNbGOIQ+Prh9qLyZdbFdmVqdA2mi/kfLE8ykqhJNYVES5f5s44/pCksTxUKrMT5OO7+fXq7xh8ad6nsuuE6GJKAPRZrrREHj8MLRfjiEAuHuwdYEktUDVuBEigtZfXXhSEhFLq1tERGp1EajA1Pquz6uyhlgiIGYRs2xfbT/5i5ju56e3E6dxvUViXFz0acs8yDyWdOjWUqiOLjrIvLaQ8t1+HD/MV7v02WW/3CNEv9LDONBnFzkrB3Jfzp1JPwhLEuvBk+Xwwuj6zdCpjXFfHt4JxrTq/HCICl5fVEy6nzQLy4THo15fqFoILAJisnB85Xk4+SdTWECIcHfRTk0pTGaaGgOvDa5FIhhka7OKiLXpG9uIvrEPLIon07SChFoWkiY2j10N3/1s+/kx/viXOt3NNVfrE2PSlQ3IJVcXzsyi4iyF7Iec0vh0Vs+5sPvwhL5oOTt0dX3b9Tu1VMcpxqNnsZBu2sepOJT7w6TUXa+rgWevjDkPul57Suyv1lvTw9xtRvR60B1+/Q5ndOnD+cN3ux+/TilRNSqgzzM0eY7o2vhNH2NWQEQjouuYsyV3LwxtHJOgLiyvUEpjZHz8tdaWJOnhKQmUuVMLE43qhIbmlEC7TK/+Yjd9y2/+cRw2q5QJj37T+/2ZJVTSamsPj+eeAg33eg6fZp1qlBpzissNd33Oq1NacwjLPY6PkiM7aFmL63zkrpdY4fAkh3O5urZOwdBaY7VNN9u+hj8e5w9grHWl4Ua1qCI+dQ8Pc6ANaakLwfc5rj9vLkbEczFTvTFhNfVUk4RnRo2aeLDFKVMtrObLrOl5z/0pg0NqqY33Yp1ZqAhSQp1Uq9pN9+pfl5Gn073UG26GlY88U7tN5lxr8esL5A7V9XgqOmqeVVg3Oc6M8zhlKvbqhZh8u4bAKuqJsYqUU6xf6HrAPIWseC74cJjXlksRkt0L7SnJvR91VeQi+7ZbPR7KbbYyH46Rx9EDk+oaKguDtTVnLOmyUAvNoh21RhfbbtOwMtKTiJpJrY31unByHAFVLIzd52Hn80CncWUilkRrKpJoJhEQjeJiTNefb44/nk+/mWc76NW0/47FZNWJixTnZpeTOWd2mYisOS6v01r17ffpcPD7ffj7np0jTi9frK9f5ayqlfPJY6r9lVRJh9FYz31O41HHyqfD2VLfPch4f85at12s+tTrObS7O3Pvc6/pgDpTQNMsEIi16bICaGxpUW3TeEC8ujtFTBR5YNcrIycPD7aDuhCg2ox3OX1LavhYW7a1QkQAMDFdinBVUVjtuo5zqeOoKa0/6zGc84ztKh1kZNXxXUwlPGJ8wpAQlmrhJOQaq63lmkG8fL2WXE8n3t2l6dR9KPL+IQyPN5ddn8FiPIpM8XSsY5G1xqCzWMqrfK5R78oU1Trt1vZU+BJ6pH0dEobNlITKmkzzD9G8DTGe+bitlUwnpJU+EdE6qNqgeCLoTjJUFGDLd2j46nnu1rLlD50cAggVjYi5Tp2qqUwLw0UE2URcZPXJcPPz/M0vD3m/2gxzzK5uyFZRppkzfQ747J0q5/78Hc+53J/rxRe7z7/opie+fJujylTwzYfpdNKnx5rXIqrrydNo9VzGatpje5NUqlrXjfYAT+xqmd/f+/upxlomuHm9FuwxU/rDfp7mvM7PVMBFDKONCClogzQHg9FIUZqS5i4DUmtJkmwZfyC08bwZS9XU5o7xjP0BoG09aFJiQvsjQA33oqbSZQkIDF4lIJd/lu5/I6e9DUmutnamlseSB6YOVrCacUY+keMTq/rVzbRN9vjttN6tV71vXtKoXiAref9Onp7CS+Qehd1qi+0QeYq+z2YaJQ8Dp+mQU79WlbXiOO1nY3f6NJ0wy6hUYYfD/f31u0deXbgl0DNZ2yTPLPOZsKQCdydJmqp1vYiGSFaTZIRINCqTNwK9ats/1auImJqwLTkXVpioKN3DHRBTYXUyoEkBRtQazlrNuHu93lyuf/eb03wRm2uNsc5j0a1uL1dZ4GUegtPbGfTVhW6vtqa8f0jf/s6vb2S3tjKPDFxsh3pG2buPCPqcyjxI3wncxpOAqUx1tfKuT+tzZGK7HqLMpzy/WA99VVeeylljEKb90/z9t9Offb5VCbJCqUt+j0akbMcyqYpEKdH16IYsQNQqkhLCxVRNSZQaz5xFqEGSLWINwUfCSDBMFASYAJ9rtZBkFrHMvaO6ACmbJtcB/aY/fvd0VpdLSRWKjtWn85Q2Xb/V3mZqvruT0Die62bLefRvvpf9g336xXy5yinbaqdCjar76TyFmw3TCZy1VpTixTlOCToNhiGnSLSMIWXTci70SUavc8AIVfdZP3zTn/cx9FJS9aqaBUS4i0owzAwwIlRsNXR5DRUg4FHcS2KnoVShqaFCaR7eQBcEjHABuAyi0QatIIGkSWRSg3sTdMhzPRVBBY0QSVx/2V2/WY/1KAXSZelKnTHGJBAtFrAsedeX+wM/PKBsLXXeDXOZ03iX88o3F91+LrnXqy/L7tjXfR3PZazaZvabdT/FOGB295KyZ8NcvByPU9Qpf+dlqiLZ1zpFmMeqdvb79/rdXfr8dc2qwjzVgKoaO1Eza5SkFKt8yYuXK851fz+NVZVSi6e7rz9kQkS6rdkqd122JKKEp6ghMFOLqBE006WNL4SIV9RaYFDpFBbwoAuETb3kMMvhvv28e/VVfv+7VM7UwYc1yqPNZ5RTqMFMzhbdkC92eiCpLJXrjbIXVE5TXdOzse8kdP3hm2l6ysy+vfa+S4fH6gUC2wrm6B5n53zU0KP7d+cuJ/70Ex9HzicM6+3bB3scfdPX4x8Pb7/evXrZXe1kFvIglcjrDLTqx8Xk8nb14svV6mWanmodn/Z3ZUZU9/TNP5yloIbrCqmX7TZtrrvd9Wq9g6XcEKloWuQtbCwjrcF5LuGRcpaFzhnhrKwpJTOi1AiNmvJl3X1p736t56NvLqzvKMLzcZXSIJz6rGevAtmsU7ZSSh3PkjpNA/04zYVPT7XrbV/z6fF0eKzd0KfBVgNQo9OoKRiJHu4cJ09qId37+4dvJv9skM+v+sdz+vvT0/vjPJY1L9Luk2F6K1//un7+1e6Ln/RJ+fhdOe5nBZF16HNK/fqyv/5kO6ySuvabvLo8P749z3UGkQrHElMIbU5y0vKQn97PD5fTizf55vVN6rr5VAhYMicN2ib4EhVwTUZh1BmWWBtIlVjY0OQYIZy89NfrZHPZn06X42rT765F4ILRk1tOl6E2yOEkdM+WJnGEQ0oeujKnw17G95wdWear6/zqSyj6+aiHcbakCieqdzUKBwYse+FF1scUfbbfvvWv7zkXnjynUV687uZUenaP78p3341/MVzdftZdvEIZWcep67OlpApNJh3nSjtZXsVw2UnGeJxNNYXWlBCUlNUTa0yAnfaUAxXH3RfsemVFA/TubqqiCikmJOG1UilsTOI2aoQDKslrLSUcqbvMu3+Sjv85nd5Jv0nbnW6GvpRqKqjzftbdCyaK7C3vPJ2jnEwG1EhzLWYppdqvUpc7eLr/LrpVPTzJuUi/spgKZ0W3sjx2ue7L6f3JetMrjSvzt2e5m+oXcmbcDkO8+TP75o/9jCrsxr3VuZPQIdsqk7v83MuiF3KEIGCTK/pVPwxJQmmaFo50EAxrwF8ckGmWb35zd5v91etLzWCNVvKUGmYiWaGso89hqWNKCqEI3KnK1JuKOABXlT5dxCf/egPlw19Pfj7bNVa7Yb/36THUfDpQigydz1Mw0u6yK1OhG1wudvnySq5frFdXeved/uHX5TRG9WCwOse9MQIeMqsY1qL1FFmQlNeS1xXzxPDxLW12vf25Vqz4FtLlLsvrq6FbCUJZG/x8ltfKRx2PhsFDuqTb62xv8zRpWkpkW2S5rT0TnKMLOae7Xx93q2F71Wt4LQSYzETgohT4xMOpWB9DrzDPXYrqoqKp16RmWrLnJDlLfpX0X/Snb8f972Qa08U6LMl5qrr2vF37uXpXB+vnPb0/ba+63YWezzEF06orsPqAeRyHlcgMqF+8SFvH3V09T73l4rGfq9QQJl11VPNS436OD6OGp1Hs6mbVpavhQ/z8X9mLq9U691/91+vuAjGhFGjTsnOhP4oATRxDYSDAqzerl0/zN18fEyTIaFxXCJRpOVEeqnF+OH/zm4cvf/FiMyRzoxShk2GgJNZj9ZMVLbVGTn1UelAVROl71DJ7SLYsqLXm7hN7+S/L41v/7m/H/ef40b9YrQXHqU9aerHL1+vxOo6T+4dht4mbz+z+nR2/53mMx/f++LbaCtIpZ7ireKw7vd7Iqp+OZ1oVmf2B6S5kFX6xTpNWnFep+MsX3Xa1ybvLH/2o/+rnuHrRidv6cvvZV6u1ekRIzpWubaGCS8N5oVtFuKnoarX+/Cu4RzJLKhUCsY/q6cVRYI5Sifdfn1N//PRn6zRAPHsQlE7Qr0VXcvgQlamT07qX80nFkLMiqQhiVopD5nnGNMps9fqri9d/MR/+H+O05wcfV0x4YNoWWjqexDpmjdG42u2o03g/81EPZR73GqjrfpDMuYZXrzMmRkr9PPJ8nqdZEhIG9VN9jLR1dse8cr39arj68YtievOp/Oxn9vKFTfuY9rLuPaUK6QlaokQoFIEmhHpW7i0gnAEX9hfDFz99mdqXFUAsHFy6N6VOcQKavH7/m3eql7tLE2aIqWBMkCHWtzb9+vDwtmyu+2MqQlmtdb01nokqAsm9JbPqjogMicTbn22Of/Bf/cfjN38Vl8M8wFYb6przwbsCOWIu+vgwzu4U271KZarzrHqhzqincznD1K5f9AZ/fKzjycVks85R1P3U+fh6kOmc/3jeffXK/uwXlzLo5rZ7/RlfbGQ+1PEkUX26Pz+ol4u1pZR6CCSllJJZkmA8qzQWGQLDHRFiq8t1YnDhqLJBJVT31p6olWQ4HDPe/+EwPSYRDYKEJe13oj3ffKYP7/39u9pflkvjrFkMqZJD7TtTl/Hk7q4Jl5e7w/5ctv75v9nUR37/2/3xgt3rxE7bdZWkxbXs/f39ebODrnsbJrj32+SrVZJZqcVUet1dibi++35W2PWmktgX90m6tOpX8cfDag/r3wx2vdrt7NWNbZKdD7W4RtFSXU+1Fh8fq2XRXgTa9Xl3s8kXpgEVE5W2wRr/KCgaCIkUoIAmwohniUSAqDVEFJRKatKxlK4CELrQwajjKelaX3zOfyLb//Q/HB7uaD/Wfooape8tJ40uatXwVIvvrlf9TtR6jzr8ZPOz/2te/z/tt3+3P7x3eO9RrKvdtQ5d3Fz3s/cKSabrXZ5NO4pcCCKXQxr3IQNrqj7FsdRzsVRB0Q8x3hVcGt7dR5z56Rfp5U/SajNeX/cm9XRMpBBFAAorQDc/nJNKiDFopqfHcfvJsL5ZDYKEvMzpGfrcoY+IJItaMp73FkENB+mNgBogneFKQTZQQVVEmmqJe+kv9NVn9Ysv8Vf/zo+on7xer9eREqbJodr3WqeSsojMcw1Sk2W3evHn9ovXt6/+4+a7//T+6Zu5iMpVqgfIi3SxU/Z2upcyupjnDNHozatYSVivLW/k+HA8ftCxyiFGv6+Doc7RB0K0jLZb57TLth0u1zmHTHX2CA2tYJ+TmYrQbJkmt1mfh58eMB9r+TL0dms5LSoXCJoHTEDa8LA16VvjuRY2WaIQTvcQiCilWX4k1RBAlZTB5Hys4wOGbXn5KW8+59e/LO/P++vPrXgX6ARgLSpdIYOIajHTNCm7NM72ql7/H1P35e3j341P3/rh2zFqYI2ARZoUyVwfvz+IJdNeZ6+1INL6NlKSp3f+9D3VOKRuLH4XZQYIzBOzWv/J6vJFusxcZR/Hyb0TcwpySgTVNGUVuDZrIJY2QXYwJtt/c06w9NIsG1ojjxJN06RNtUA8zxcXKlu09ipBMQbVVTrpVjp0yjDSPKJMc+o5ncu0l25Ib36OX33v331IeokJRSyy9U0ONAywTjTpfJ6nc7E+QxlPhlOfdnb7v+ku39b3f4O7/5mH3/hdnrst8+bOuhXOXaAWPY+n6LNWhjgKXXq9+QRP09mPq6fj9Kha0G0mySbDl7vVy/z5p/l6B5/nMpuqNHRoKpo1dSnlBLolc1ajBAVNmyx1PMTDd+f1rtuueoS6S7N4SarVIy2SwB+kZvyoQ5WFDGdR0XV2ebNZZZsnlgJxFxjD6TKPHlO9vZB/+pPhb//99N2vpjdfrObsx55B22yiXxmrnx/H6aSuGXkyMR+l7BmqumL3qX3+5ur2z/346/PT7/H+W5w+pHx9zutI2tfjTBpWppQ6+nmcQqs71PTNbR1ke34oY+XLndz85KK82r1ayVevLGVvUi3Lz+05Re5S16ecDDRVaYL2WiIkmr42WMfjvL+fthdrSyCV4QuHAUjkwkJ6ptezdaabQEckMSLo3bBarbveFFLFoLM7VDVHoMxhlZcZ//RHHEf/m39v9Td1d9Mdz2dWE49hhfFoEjX3fbeSzOyQOVHXEHQBz0ja4fqfl6ufpu3b7eZXT49/Y8cPaT7NWIcNHLZgH6uTHVkJjke+P2GV+y/enMc0nKf0ao3P/+W13QxXnX3xSb5YxRRSpYlUkhiTaTekftWlTrMZ+VEWW6UZt6iUMiNUz/743fHy5Xa9VYEGKaJkEEgpJYWrIUIlWfKolY4QWJPFZK1VOlmJaHAyYRL1lJOKiOoaQsrxcWLg6kX92f8a7761b/4Qh8nN5OizXotLKpHyXPMLyTtwjzoq6JKhHuPZUx8pS4yZ0M1trG8vbn8+n37jf/xP6f7ryUtMl/PwopjpOJb5RKOsujDnH79NXz+i79df/iJt36yGlN+8iN2O7oCIWGNxpC6zH2xz0eWViaiYKKUUaqAUB6XVhCbmwMwZR4xPx/Vms8xMTVq/NDXPL1F+PHzPujMxS6QLzcz6IRN0UpIZAdUKZBOC663QcT5MAb66yL/48zq7+Tw81PPVmpY6ughrZcxzsdpX6lhnUagZKuss5+gsh7pCQhRI/XC77i7m/kseft19/ys/fn3035TxzTxZ9zSeT3NA62PVdye73eo/+9+uXn025JW+vJKLdQcPto4nkLqUVLvBttf95mKVkolKBKNUhEPhIr5YhXkTY6jSi5+epsuXK4jI4ogGM00QUbNkCAeDrmYqsGU6IVAEc5KcRCTBNITWddKArlAUWTxK+BwzVzKfX7+yL9/M9x/OKfvujfTrdD7WPufcKcTqyceRqUfO3Tyx63yzxfnE8WQpQTQArLuVbdyH7ur1zdXr+fLn58Ov45u/4je/U/bTdR/n6fh+yo9zd30r/7v/w9Wnn3a50/Wq61OwRpNuWkoKJrNhbeuLfn297lfZRC1lEj7NpzhJCRFxr+HP5zEWXlUtAqQWuYPRRj6JdBIQS0ndKeoqy0JA2VgQmpB6iBlnQNioc5oNpEmYoJstTTHP505x/Rqv3sn8lK4+6dbXKQ8zIpfiuTeGT2cpswLS9VKq98n6VXiEMEFdk8HD6eKSc1HKufYycP2z/pNt78Px7S/r/CAu2wO52aX/8//l4ic/SixNpVQjEM1GQtEzCdCt0+a6211t8yrnpKopAmIQFa8x1TJ7NGF1jQgnqSCjNo190yipSfsWkqUm3KeaaNMfCwk6KylEhHuSyJ21Obd7rSiqamaMCGhIwKgqSaxGdJv8+kd9FN28CprX6urJXU8j53mip+bvl3umFKKohapJJHJWESAJVMLDmRxRJ8I7subL9ObfXF/8uP/+f+LXvzz7SX/+X69//HOVcyr0Sq0FpFiSlL0XHcxyn7a3/fp6lXMnz1IKUU+pjtM4Hqa5RqmOAKp4hdcmCVQGIe6soDJcwCZGT21s7REibqY5de4BCTVrxU1RiqnQWAIU0ebxYirwCEGq01RmjwAqfJLc2/WtHO7cZy+zMStKWNauhzVumHvqu3muaq4W6lBRTZGSkZ57TSa1apnhNaqHkEKVlNNuvvwZhlev7vrp/PfjV19Zp+k02tMhTnOMp4jAekjrLbFLm2sMN7p+sbrYbEpxMFKXRUMTp2M9Pk7HJ6+NDtrc7YLuiAiErTrZXGQ1skqEf9THNmIIRUQNZgqXuZgghPQAAmJGhZdSZyGzdUnVIqqoBN2rT8WnqZYa81TL2S4+HW4+Wx3v53//l996Si+/sMzSFaxy7garKj5KLS6GnNNcaYl5TUDUmLtkCQiqaepCVZWMQJ1oMgyQSWQ25HX96iebF7d6PuDprtwdyvlId5WI2Xz/qPpZfv2pXH+62VwOVt2SUgN5ns9+eKjv/7A/vZu8BhMVRLOT06hBVjGX4UXeXa2Smi9MCFULEUleo0bkLomIKSNHSuJJqiMY4QhIkPPoU5rDQ6diXVaFK7yW8TidD/V85DgxSupW+fJVd/Np+uTLoufh668nqt68ma2GPUTthFVNNamwqoMuHLKpSUoiSk3BUEZDQAENViBMqsxTrS6ch/EP5VLyP/2L/s0XcTwcXTAVjhNEmTPD1CMFknSSOxq0ssznqU6+P56O93X/GPO5MipZmyK9cYNiBkNQZVBcvFjpyqLoc32IRtJNUekKs0BquyxytlL8OQY5CHGv55hMyCqq4hSAHqWUeSzTycs01Zlk6q9se9mr65tP5Bf/1frufbz9By8n4sfp8Dhv+rzJ1MG7QcS6lCHC8JqSEilc5rEQJlAStUQEa4HPjooOTGoY8PnPu5/9m83lVcR0rEcZ1rD7RHfAqjNGbnuuegryhz8WS0/jXMb7uew5e8zTrAiCLkEGfJGFUxieEMmibj5JLz696nTwIhG19fsoACS5RzMJkWZAIZI6TZMms7kEIAqBs861zAYSFnD6VIU6zqWMBYVKHbKwl4vbdd+tUGJ7dfnP/k9p87n88t9Pdx/K4Y/jPGG/4eY2icZ6pU4lLGnMxcX6TmOenGEimKeJkYSasuzWxJb9xoahyxkcJK+6PJSyP3z/97CyGtb7Vy8lQc8nZxXN6fbTtLrGdPLT3VymmGthlQiGOJoRDkSogAQdSISAIQjxuLxMn/50u9olL2xGK8+aWAGQSqlCtWo1VEPNhtxNajCDeuuwSsoG0KcILmwvVq8e0xwB6TpkKiKvr3e3rzfZDBUQfvpn3eufbL788/j67+W7P/DpCVNBOZf9k3874eGDv3xVdjvrV6zR9QO0BCfVzkySddy8yOurfrWyLrkpRZIao1IE4hY1q6eI0nXp+hbbLc5jJZAHXlzZ0Odynn2KudSgi0IM4g4FKQyJCAgjlCjhrthElfXAz352c/Vqt3A6wGc14aI7SSTm6uY1qB60ZGLM2WoNLWKKACA+zyyFJqKqc60OAyHOVd8NGwIh6G7e2OYCaqNrxITju2maJj/55YuyvrC5uIf7hNOTPry1/aM8fRfjna023WpV15vp4jpL7/2lXd/u1humrCmrSIhYm+CGt4sPQiCWVtCDJ+2ZouvrxYsOUBUDPOZaC4Ph9OdfazRPD/cIEWiEkC7ixuxTbDbpxz978+KrnRhqYQs4gWWcqACx+HR6KVKLJMsAzDTnNM9hxqqeVJMJKBGoESLh4RDtOxu26HvLKZBMc5+uchGWx/nx4bz/rjx+OE2TRyGSW6aqmMrQpfXn9eXn+Xzi4SHGow993l7KxXV/dZuHbdevtOuhJqyBBg1hHs3rAiIKlYhASuub4XgY55nSpLtR8UwWixCvNQLhsdBim99fLF4Ui5GIgLVj1SHzy59dvvqnF7CobYClRASixc9FPJZaiPPqtbj0SSBUsxS5Q63qFRFhIhLCGklFk3U5GVK/sfUudevIQ5onPt6Pj//TNM843J2Op6mOAYYlQAMOLTAVURlR7YCUzp3Jm08trfPNm93uyrKahDbXyHAoUjR/CUhl45upikalgsmS5qTw6Wn1cDdH1PAUHoBVRCz6tXBvohwGfaH1NWGvBMTEBSUr++2tfv5Prl/+aKtWOHYMSmpywqaxo7IFeCRSQRhUm5cwBGrWo6c34I8s28te6DFpMkmbzN56S2lAXptkTid///Xp7tvjeHL3cEYINIFaXUS42EtHLIKWGj7XALGKYQj3Oqdu1YdUBwNOhEtobb7kEaHaRpoRpJqwuTCCw0qu3mymMQ73pVbRLrc1IlFL9aAXBqGiEVSRaAWzKOggpUrX6avPN6+/2m1frkQC1SJkMcttGbC12hclNBICH62caq1WkyVR1ZCccqzWGF6sX366NaCcoQpNEkly8iD3p+npm+n4vR/uznX2SAGbs9Qm/48AQ5tu/aOSDwTZbNNYTixT8fqgJV5+8iJlzlMVqLYWiGpzdhZRLFKtNkQQCYBCSasLffmZdyL3D1MoKcbC8FpreGhzFSoeAonm3yxCQlPGWIdt/tG/eP3my41q9lrcDVTC8SeK8ee9KM8IHiRZq2tlrdYh6DWaQ47o+mJ48eZyfWUsVTuJnLrmNKZx9/vj/W/H412Zxlp1skGDk0BETAhSU8r67Nu4VKXtVDib4a3HpNDzQ/0+QrO9/uTCLEUAqBRpLikMMGpTyIgiKhtnH6JeRLNuXqxz6lJ/OB5OYyFgQaiQolQ+uxEt0onGVZAOfbe6fNFffNLLkDgnYUZ4MIQBQZvh/+BU9vyRnGQpECM1XMKZu6gIMRFZDReauxonERgRcnaRiA0Of5jv/u74cBiLjEgKQdBlCYWq0i0COwHZVIr0Vn0JBEL3RluVoDL2D8V+e+g3ut1slu62MKovTmF81mw/u+UIAHExgAxKd5Uvu3V/SI8fztNhVsKUpUYEagmxpgBzU0mmZiqdDUkR9cNvD37N4SIsi6ZgMYhFOH+QiS2k0ba3UssKEVFL1MRSvF+ZQEEMg2x3KzPzWkjJlpnDuhj3p7tfzk/7adaqycTJgJBi1tQZIjBDSqYKirgz3D6mIUbgI2GcDHdnfng79htZ/7xPWaNKVETQmttYPJv2Ne6mWo1ApSVbtO/Kbt2vNhvT9K4+eHiCWY1a5VmB1CAAky1VESzOo45fH5/ejpsX+frNxeVFjyy1cHF4e9bgt5VqL5EItBlPmX3OVWfp5jBVkppqymCIIgXgNdLg7nj32/nu7qlqUaU4GVXNUkoeVVRNNWdLWXOXUlYRHg+1LpzBIKGqDDC8qdcIBuYylfs/ysvPys3LAY7wZhZJkqri4dpsgwQKkaTwpp4nAyaGLgp89jIWlgivZy8CGqnPb1bIYAiJ2tp0gJQ6z3E6T/Me/PHl1cuVoUa1/6U0n4sWAEimqBFkiNh4nizZ+TSu133qUrdbMSu9fAxzqjLu57vvDyNnzRQCIUzWnItzzpaRU+pyyp3lIVtuLdbp+BSkAtWjpkQP1ELCCURoMCg4HX1/d7y67QCTFqFQwbZP5fl0ttDXJNzN/QmUUPjp6fT09nR6HCNYJnGn0ClQ1fbrZqgOBaikEazhTiSOuJ/3NWbo9eVNj4Sof6KAXqQ4IJma9yMD4RGB42HykrKm7Ser9bpHI4sGTc2yAnr/zePxcA6FQQQhrVshKoLUWddr1+Wu61JvqTdVUbBfB50Rc6lVVRjNQ0l0MdB0mBGsZXp8f3j9xdaSqTU3QQCxiLbBhiprLYTklJqvR2tCzvv63a8f7r6dainhEtGgQvNkXmgx1d3MTKU9QiI83BkBSzZVn7/FVD/85Be3l1cZKr5w2dm6zIvAzgyWoAYPusd8rqcnPR9hK00JUQK0RX0vMc2nD99MtYgktC1t2ubXSFlFmZLlLqfetBNNIjmFiXXQFLmLlERgzbHwo12yoJHsCZXTyGlqDRpfqBnLj5BsBpNYnhwDslkJGEXiw7f7978bz4e5VnpMZHmWxfGjJ77q4uas0lyKRZt1otemBz3f+bs/HqapqhmbWzx/wBENnYamUKMZGER4meZxoqbOlomGamuoJo5TGaeQrO1ItlSspmrSwpYkMxP36tW9BgBLKaWsRrNmkeyx2Il6Ewy16ZIFRCXRgBAlwxcaeTShtyx7i8ubXgRpJqL06g/fPk77SSSCBc0TGaFKS2ImZpKS5ZSSmaWPDimiYounGCJwrnV+eDc9fTjDXKzxPhuihqmqSiJpaproTvc2qKFjE1LdZk25VjeIGI1xeD/NpWiOiFDls1kuAnSGSG58b0JULWfJw6jdMmwi6L4YLrlQbUFBjTTgHlD3MkGqmKKSQRPEIu6Q5hPNpu0QkXiW9hnHcz0+Voo/B2MDvd3+NiuAqCxbSUWqiNUaDDh9eQBLiDNUfTxOj+8OF7fZuszSLMefXcsEqczQ4OI0DajoLHI+H0+H1XqXzGsKQ6JY+FzP7+YImAad3h4gBGWECMOlzK5JyACFjFIij6ZGy/AoEVQxSjTLTGlK7GBQTBVIImYpiRrZTJaqu2vD8SZctO5LipKldaKCOp/GOmsVaoSoRFQVqmbTpEpVUVNLmpICwugWz+WAPC/Fgn2IcD8+lTIhZw0uQ0NVafYYqRZoqyBo1mYQCfNpfvuHeXtpG7i0FCQgRWrK1BBvD2gKhqgKhRpofnhQJ0WChLMgchL1AcHwWtFQJJjMaqMILCVFBIkSfd/3uW9PBBFVr64imhrvonUghAuduEEoCDie5nEsIU4PbcJnTe3HGmdm6Ps0LC/hVd1nEikpQJ+bhWzLsk5a1PCpcNOJakTTtpoAKWkSNC6lNNBXBRmA+93v969eXWw/U7hEASKl3je3+cPbqGNjLTmBWigQNUi4pUaQ05bqGJytuEAna1cuEmp0b2K1Z6cWRiCMOVP6nZkZm78xVFMb5rZI+2zB8Ax9CJoKvI5H96gujhBxz6bNmbUUiOqwsc3WOBCzIKRIBATRajB/fhBRg4CtcqjwoMELIRSV5REzrWcabRgNEm4SatCk89Ef3p9o1OwtYYXaxZthWGmM4u3ZMhHRZlvuEYxwutdalq/QI7x4qbU+vz0BxNQg8Cb3X4oJZWXqZfeqB9G6Ci07B1uzJdrHgiKwbPa2bLVgmWq1JYyIpU++JITuwlabzjQx+JxemktP41e1uTTCES0aGQ1h7YRKUz8vT7iiqqikFgM1WVKqQXV6ejgej9EM5EmGc3UZVy+24im8hDPcl3KwzT/cSdZavS5SHnq0ZyehMeTaJUqj3MvHlCwE1LYvtpubjIXfs1ykqKouXtfP1MXl12QxXxZRax2O5ivfnmnlHhEOl0rYWoZV7rsO2vxIbUmszcmiJRBRwEhRM8vKqFisgReDFGdozpqSLMTJZifcriPbuJ/3d9VRLFMktArFdm9yl7KHUkwkkwiHV2nDGPd4dmHn8wUtiBKqJBrWjKCHC1u7Kglzzvnm9XrVpYAhQF+er4amLlURyAI+4Gge3SJOp2K1prGLuqQRwMxSEGoWBE70icgiydWgBkZIiDhBKrTtOAJQpyP1KaU+QmAuHxMKKEJtyTUltaTL4zSapz9YjuXx/dmDojBBhEoMF6+2F7e5nlPM0cx9FrM8otTqHrV4O1miKoteE89bA803qEVrFTFLCgW52ur6pjO3KOEBVmdEozE/1xzy7Ja/VOHPtE+9uMrdAPVmTqEEq0cEa2WwztP89KHEDChIlFLd25ReBIIaCICirUuvHAbLOat0JGKxAkQwVJiSGVKIVBKNCiIipGSNKXR/V+ZZc6IQFeoVacCLL/sP3+exeiQTgIz28InFngvtZLouDzPThUPoc3sMXos/AAIBwl26bFe3q/5KnQJIMg3KUhA92xU2A8KPxZoKVaGSQe8v0vpGnn4DmlhqVxSatNaqVivx9L7m/NRZarHPwRC2J2p8VElAg8zrXlcrBbxW1Lq0SlXVVPYz0kMablCbiQSI6gIEKAEB6rT38VQ3l114SAovFJHrz7affKh//HWcjiWtRMSpLtIhjEFRjWZxaqGigXAPTdaKtPDnKp4BURUTxc2bzasvN13K7qheiMW1MBgSDSvg+blL5HO3dHlaBixv1q++WN39YT7UUVMwsjRvbSBcYeKlPr4r3WDh9KqBqLXWWqMypIU9QCTNMmz77dW65YBnq0gR8Djz//4bpP9chn+W6ksd3enuzc80IOEimShSTie/zJbNa6FoLUnS/OqnqzLGH3/3WM5MvYkBkAhXEYFSQtQYCGm2woEKUW2q9tbbTckMWVzsQm5/dLnarTiJR0FrOGDRZj+DhgWPqehCVGstZpUAk9mLzzbXL6fpfSVnQbQIYGrN11F0LrNFy3yhzTYFNJDOIgIVFaYkcfkyr2+GEI0aWLQ8GCv+29/hP7/1/z/zJQhk/KyyHAAAAABJRU5ErkJggg==
"/>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>(100, 100, 3)
</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAA+HUlEQVR4nE39Sbckx5EsDIqomrlHxB1zAkCQAAiSNZxe9apP/8pv2f+r+/TwXtVjsR4HgEAihztEhLubqcq3sLhk58EikZk3M8LDTFVUBr38P/7v/4999WpVoXXrp5ZNbZp4mGtJRSAi5qlOtfSe564tBGPxgsyW2MCYduXVrb3avfv+5v/6f7v9t3+9O1xpSX/uvUXJLqStS+stp1IOkxfL9fn8+H77+Ev724+n//rv5z/9sP7y3EOIhHqQObmuDtPrd/HNd/ff//b1t7+7+uo3u5u7UrRLdVEpRqDXMKeBnuagwZYtfvnc3/+4fP55OX08f/64/Px+PT5z77yeZJsfz4s77ve7OVs8nM+Py+m8amuH3TRPpUrWI3tE9MhQShJA0ty9QFKPMPbA0mLrosGMbmZURhgBWohdCiFBgQAJC8+Eh3sIuXR+XP78Q319O3355aydRaAHsqcBxa0W35W5ZK6n9dOH9f2P689/X/7+/vjpcd26jKDlxExEpjJxPvb3ym09PnzGh8f23dPhu29v392sdV/MZcxSiypSkZFQAgRsmuzt2zo5rg72dIfDvdt+/vnHrT+3THqlNjRxC02GelWDWpFLj+fICMyZnnSazIAkkqK5EU5aMZCyDK09logkq7nTkOxSAKJ1IFJraOlqCRaWAq+FRbTarS4hbbE9tP/8X6dqPG23169tswYkQS8+FZ9qrbTlKT5/2N7/1P7+fnv/8/rhYx6PNCv313Vfsa+2nNvD43pcMmXb0X5e1oenfDzGp8/r8wN/++t8++7u5nYqRdMMem3BUCckqQPFbJ54d895qq9eXZ1P++vXUXbPv/zlMZ5bQisiGovH1XXd39Z6PXWzJbUuPVO95UTszGhGFgCE3J1wCcUAgpFoPbvSq7sbYRmMVIsU6JEpbS23UAdJY/E673xGii14TqljOer0t558ftrw1dft+jYP+91+nit9Ll6JaNvjw/Hnn87vf+6fP8fTOVoz0K92fnvtX91Pd/v66ePy3w19Pa/oSJe4tXg+9o8f+g/zaj1Px+Obt9vtfb27myrNUJoUSEChSKGEF/Kwt6vdfH/Dw0FesJv6px+Pz489zlxSpfdear2yMtm81rruO7Y0y0QoE3Q6CaUhpYSQEIuUol1+hTLKIAiRisQWyJRJXcpkmMuJqdp+V3Z7VLWea2ClGwvCjkeLn7fVThv1nZerHedSdsUnY7T2/HD66afnv//9/PlTrksgWd32U9YZb2/Kb97Nr3Zlzvz4Cz+XaNESdZrqfF3u7+f76/0MOz/zx3X5/Hl983b/7qv65i73e7cUTea2IkPRA5YkYSYvdv/KZPv9Tu9v+ONf13Pi1PMMnIU1AUQW+qFMdIPPZWPr6j1TFEUm0FoaaeYlFD3YEyScRCoYJoSUyVTpSogpYapWJlYvh1IPO7kdQ4+BZ1mv8/5QSjGns2pVWXsxn6/309WhVpPWfno4//zj0w8/rL98yt7CkLtSbZaXXhwzWVymLIRZmtFV6La78rdv97/95u6r+6vr2cP649P6w4f+y8+nDx/ad9/OX7zdTVW7nVd3sG5Qz0iAYprSRbfbm7Lzm5sZu/ncC1br56d4Dn48Cq0vIc2+KzZZ3UWJpeVpyXMimGaR2SMnt8lYMnONHskABE9YpwIETW5g0BzFSLN5mg6HcijT3ln8uMbjpseG1bS/4qu35dXdVHalHHB7e3j3qr5+Nd9elcOM82n58Mvyw5+PP/x4ev9erVut1Wyqbvur3Drb0nqP52OX6+nUtw7AvUzTXN68Pvzhm7t/++2bV1fzXMvSVfLh8dPnv//y8PEj1uf5/Oubt2/nu1fTwczciwySaAJBiVLC3fYHM7v2eo1pX6fj+59O23l7OEd7XhN2Nc2HQznMpcpjrXFke8J2Out8aR80E1i2lCKVLjjNIQupp4Iwo4Nws2kqU/F5tqv97lDmmWfgnPbcrV/l9ZV/+fXVd98evvnqcHs9ceJ+X/eTvbq1A3J5Ov/w96f/+tPpz39pDx8S8qubctgbgy6vta19evigtuH5ORrz4XnrIZrX4lc3+3d387dvpm9f2W4KgIc6627+/HF6//Pxw1O3pna081OcT+31l7q6rk5YIpwwRzIjpHQHTMX9hviuHm6vy89f7P7609OHvx3Xs2fY1TTXax4Ochk6cbNb9zh9iiW1rkEn3DJVFjEJMzNYMXezMABKgbSYjNXr1VU57IPUbsq5rtxOka3W3d4PE1+/2//226vvvpm//mL/+nBlk3mVqbXl/PCp/+2vx//44/N//6W//5gQv3jNV6/ssON2jgrtZjsvvkx17ciOc/ZI1LncwljneefXE15f2bubYqZl2Srifhevruzgtjzz6T1wXrfn7fnRzmv/4ouru51jKsagE7RISOroPQOSzfZ69rur3avX83xrZgV4Wo7Jmel9UzuUuttNe9+33f6zmeJ5balgsvStlacEzapxMnOSxctEK7Sk6M1dBkzTYrZld4WsVCf3dn8137zeX19Pb+/nX305vXvjd1d2mBzFwTg+xk8/nP78p+Of//fpr3+Njw9Yelwfyt013976fvLFDAED1uyFFiCR7n59MHMuLWjuczk4c8ut9evDbq4Uc9s0S1dmK9XWPLf8uGk96XTK48f25bvD1avD4c7mitGsQuiRqTQzmUpRLSjzhOm2eL258oePS5wytv64LBvi7tp2+7lel2ntZckSiFUZ6lR5hpdCTj67sRTb1ekwT1f7ZNkaHlec1vMSuS2b1XKYDdfTfHNz96a8fjt//cXu/jDtK692edizWM9+jMVPp/Uv//vhv/94+q+/rr98WE7PtjV4yas9bw++d00h0pNal74+L7k55YpmhdPOinPXJXXQ1PKXz8/z3/NXb24KoC2fH5fTp0WnZuuaLdZgP/P5KT9/Xj78vHz4qn/xdXzx9f6t7fbXXgvdjckQEyAVBssok7+5KfW7/av78sv788ef1sf3PH7SY7N+3rqjws4eeVVqHnTOvnY5y1anKKh1apP7XHmzm29vbl/f1rmclrZ9aJ8+xmPrnP3+9e7+q/0XX9589e76V1/t376eXt/ZvhhbU+u9xeelH0/LetLHj+f//l9PP/4lfn5uy5JMzK554v2eE3J7ytY7OzOxPK3rw7mfPdPSehSaGzO8dyZExMr323n5fHp8sxwm1xZPT9uHH8/nj0s/La33HkwxTfaJnz9sDx/j4eP29Li27eaLX+1vbmudSUtlbulKRSJpzM7C+zu7vtrdHMr1bvq58IPx83M+t+X0sKJlbvJA2c1SjK8qsZ9TOLMGvFsxL7t5z8PucFeKeKz6mTvvuH87/+bb/ddfX33z5eE3b6Y3N/N+nmWpiNbi6XP/5Zfz+w/L02NvrR+f2uefYz3mFJuDkIw6kNfo7SE/fQRarykE12OLU9O29p7hDNOSMBEhl/kMAKdf1iP1YX+edxP72rY4H7f1ueUakiJjixTJUuPY+3NfPrWnT+fzsZ2Pd7/69fXtK3Lqk6PI1gHzzTsjS9+jzuT9XamGqfTDLfxDPHzA4/vt8cPSV+0qridQ3IAki9XSWp6lZWsLeZ6Ure3Qb/fT7dW1387Tm2kzf/fl/ttvb7764vD6qu5LuiP6qX3s54ftw8+nv/3t+c9/e/70uWfW/Y4F9MBcUNOXXHtDNcwN+ZDPnxjrwtYnmsMypUa27GuPbEIaZLRKluJmcyJz25Zte/4sGBkkmF3ZBcGyWfQqpYgIdetrfzz15Wjn0/rp8/L3T+uvvzl88a7e39TJE8VaCABZlL6RIcAxX9Uvyqu7N3n9rv3tbx+Tj5/P9eHcj0tfok0JAGKUw93h4bGdGnqqlGlBPRTmIXdv+e7d/lfT9TcZXu3urr6+K4d9em65cnnoj4+n44/rLz8+//WHp//9l+cff4mIerMDb+brCaXldl5iWfvTEmuYW6s4PRd1qrWaYK3JNHKyymT0vrQeShF0WS0VYN9MqqHoRI/WpXQzEmmZQk4KRoQkUJnqsFBvOJ+wPLVf3i8//Lx+eH/3L99f5zf7V6/KfLsrzi4lJNkmOoMIn+x6t7tRvb6rt7c5H6z7Mad2elqWLZqStFQt33x3n389bY8RHfPN7u6r61/97urr3+2++mZ+92aa6sSiUjRVKwxs8fQQ73/oP//9+OHnh/6YT0/t5w/rT+/700lzUc3Nl4hT9KU/Px7Pi/rWkAoyq4eBnUDSsDELckczdw8U5AQ0EIKllBmBzKSMggsZyMgkikQmLEC5wMzWkoCZgTCgA+o4P+Ry7Ntz6GHF07mv9+2727fYdjcT0Z2qtLSJLGKkonMxs2v36ct5mq6v9v7Vu/XnX+bnj9vy2Jbn2JYsv/vtLbzsHtsGvPry8P3vbn///e2vvyxv7/3q4IV0TzqQXI/88HP/858e//uPz7/8/RynzYXWdT4jN/jaamZ1adW2tW3J7dy2NZVBKYDsNT0LjUyaGcKUUkYyUoZ0U4xBJRVdmVRPkqRlZmYAYRAgKch0A5LODHQDC8fzSsXmXTWEFhH52PqPATX2hdGv3/5qKjubdqzOJva40E4t08yc4cbX9/WwK796O/3yGZ8+tocPp4+/rE/Pvfz+27vdzdWvWtfB3n21/+7r/Zdv99cz5mIOEAkkQsfH7W9/Of7xP5/+4398fv/31hftHVdTKhAbbG37rR00Rs/Q1qNlbioZimbISk5JRlhxN5QIognZ6Zm9J0SFNkmAZYKyFHqMCTahSKSQGZkAkE6YDBYwmetCpJgpqWjWsyRBZUccy9Pft2if+6k/L+ffnPuXv9rdv9uXUghDRpByRqbAxVXpRt7f2t3B377R81e7p4fp/afl4Xkrb77a+X1+5dq/nt++293fcF+rpcZTUsS6bI+f17/85fwf//Pxj//1/PcfTrn53TzPVaVnWyOeus5r7emgOqQs4gymAVQgKVIyhINFWWQFLGBEblILjTE+ET0lZCZNUqKn3EgaIGPKFEEhoRQkyaqcJClARjdGSAmKDrpBDGbL4/kU9r7lcV2eH/rT4/03DW/ezIfDbMYOBBFGkUkGDUBH1IlXBfvZ7279/u20bFH8Fldz3ha8elPu7opbIEMSUr3l8Wn98W8Pf/qv5z/9r+3Pf14eHhvSD8Xvq92wt+doz2scN25hYPYmBMBqPrmKZUgdnsmtRWbMU6nMYjbV6uYR27a1NTJEAkYkTYAECKRMMtAINxKUIFGJSPXskZzCJxpYu5QwMMl0WnUrDncoQrFEC2leZW3L44N9fsDxGP/+f7nyrzTPxU0d6EQnLJFqgnrmlpxYzFWc99MMWak7D8or6uxGQ4RyhRQbP39a/vynp//4/3360x+Xn35s6xKT29Vcrne2U+vH4+nzej71aFHB2TgpCmDmRsHgIlDC2ngYZqTBSXdzN2Vm9i1aT9I8FcwspZBO0sykNKUAQgYjAWhiJAGyyzIlE4rBiUzKUqR5LZrpPhHCecu1Z7SmRGT22GHTsYcUk5d1tbdf1qtbJ1XF6mVVkwQkgZRoabQxJ9NUbIfqqNVpmdkLGaHTY/vl5+Vvf3n4038+/ek/t19+4ukU+x1e7exm4pRbPi/r52U5rtHSgF31A2EKkAQSiT7KdLQWvV+OCsRIkqlRsTNCKbg7lZiKzz6RRoCkZCuiZzATMroTWSw7DYTAkAwkLCEl8kII+Fy5q2U3l5SIyFihjIxcM4OZJm8f5vU/8fjx89PX3x5+/c3V63fTtHMiOiBS5pnRFUo4HaAkhYpNnNz28zRPhRHbEh8/bH/7y+nPf3r++w+PH348PX5Arj573Mxx7Tn1xHFpD1seVUOmJDkThQllBNK0JpUpZYuWPQ1Gs2KQtPZokRIBRYCiGQ1ws9lsKuVStAAzm0p6INMkZGg8cAML3SsS6QkzKmWEiw4WshbuJ+4n76mz22TmYhBrhLeFSJZcP/F99o+f/P1P6y/v2+/+/eZXX+9ub7GrZes9wQRJF9gzMI4AWepULLmbSgWXJX764fRf//n4x//59Ne/nrZT2EZLzFzJVjO0tlgznxtOzRIzxxsBFamEItOaFEplUoTg7uOgkBDUwgiJ4bSEFVaRDhbAIWhU+DCzUmqtNdLbFq2rR0roZu6cK4uNVie5FzGdJroVN5SCcgFmdHEuTHkf/0xm71tf0J4Z6u15enzg4xPWoMznebfzFLBFEjQaBaWApBmNpZi5wcm+tl9+ev7//n8+/o//9+e//ff69NT303RtmF3dNmXkWae2zQ1lU6WxNvUeQoARkCIz0QGHJAiFNC8kJWX2SBks4VAyIZKwWkAjgYI0qUdLEEi6ebXJLJUUIwKZEuReStlNNrkQiBZyTyvKdA0JMelJ5Lb1EElVd1gtCWNYV5htYt8yPDf243k7NjSDTdrt6ndfW6neMt0tIQkEwCEeRWGyWMmuT5/P//G/Pv4//18f//LH0/oYs+eV8VColufosUbbItrqtNlsrqS4NfVEAKIiFSGmDEnAAINoTimVEhgQ4Eal2XhUgDvcYSSS2XPLEGCGyeGFBjBR3NzlxU0wt1Lcje6SGCQANwPNE6STBkZktMgQALg7aETIg6Q5o/AMnTdYVUrLk37661J3eXW9v9sd7t54NdvERBgJM0CXj98SRp6X9sOPx//xPz/9959Ox095U/n2Sne75rLzsedp7eeerc3KMtfqcmO07KmeChhM/zyxMAiAEkDAoIwgaSBFUjQ4YQSHKmcwQ8h6qqUE1WqE6eU6g5yqgZBgl9aXrSuaelcyBt8eKTOWUsYhF6jxggQoM4Po1W2aHA6TMrJNNd27vJ3i/Q/bH2+Ob279X/bz1Z2FzJMykszRDWmlOqE4P60//fX4y5+P20OrwduberfnbLE+r8vzsh5PfelGmbkTBFvr27atGR0C3FFKwURBSTDAhIXo2UEaYICDpFKdZuaO8YMOWKYyxgEUaSZXWt/U0Qm5lTI5iMyUALA1tVR09J7JRIiCgXO1RCMTgBlERvPsIWi0h2JWrNJqFSdob15q6YbV2I75w5+O//UGr784zDdlmpjNmwRC0vgLi5Gxxfk5j59sPdoMTFPZQVz6ti6nh/P6vLGlQ1UoFDMiAhG9x5YtZEZW1+Q2eaXQk2tkdCRporu5WTFzIjMS41B3ySS6QArKiEzRaOPI9xbREtzMoUJ3eBF6QoCQo7EHe7IjoSBQzVyhtAFiS7EKQtFaRg7XQRGZHZYw6DBhd22xV2du9GP39sy//qV9/c35/u3VdY1KSwz5FpIys+SWy1M8/LJ8/mVtx17ZqxCn7bktXNq6rLl2BixlxR1ERnRlZL9gtzHawozFDKlUMGWgBHeWUiazamRmAD2jSRkRmSkYIxIUJJHFjQO+txYE6DQSMcZpmBlTqUvBo0Q5kwmBCSI1piC6+1QdQA/4aD8wutOUaUr5ZPur+frdnjf11PrjJh75eOyffuGPP6y//m4/HegwCSLIgR1Uzsf14ePy4afHh4+P29YV2drC7djW89Sd0YssMiU4VABmjpKdSMIpQMzISKaNo5FGubGA5nRndSsgEkIYzAFwaMZBggZICA14gQsiTQhilay1lj3qZMXczCC5C0Q3mcwkMI0gXqR1oSZAGFFMbrREYrAClglC1exqV1/fztP97uHU42k7M6i2PNn7n8+/fNjfvjnsKqHEy0sCUM6n9eHj+uHn0+Pn5/W09nPYtipPc2vKUpRGFrs0MhvCJQAgRZEmCpaBHuiOKsLM3UyECLqbExREA2VGuZPpl8JSUQ2REUhJIN2MBoWYSCCUvXUgZ5bd5NXIkA8Mj0xzdlHj+WqQgJRSES8gxhyW4wJAQCqc5oapWJVKoNILnAgD0Nvxc3/4vPW2wyRAFC6VkizLsT193h4+tufPy/l07qcs7VysEaClDTeAQ0oDU0iigiEgbQDdAjMBMF3odtJNCYkEHQ4oKTOSpkzSiqwEzGAT3dg6kREh0OnmPt54KreWWmMYLYwmeVjKAFO6QCSYBAgKJHj5RJFtVTNFDPgHMM1ZwAIY5Saj+rmF0GTREtA0GSywYT1GbwnKSUCBy+koy3M8P6zHh3Z+3pbzEmtXX7spC9OV4xODpFRAMjjNPCWIJJ2s9GLwYsWMIENGDwJgNStWhI5LfbO8YB+4sRhFakzKHFAVAkjAED0jskX2AXJ6+BaqmgQMMKAEaAaMykkaWcwgKGPLSAlQD4zW726zeTEj5a7s7fx0bs/LmWWFG3B9VTq8gv2stgBZnBnZCZJUZjk9tudPy/q0xrLlskXrPbeAmtw6u2DgYE2GFi64Bp1EFPpsPrlN1ayKRqlyCHQQRDfSgEQOtBPWG8U0R600ePYBAtQTW4rZiOGjM4GZkQMvEFK23pyWbkpFRFKgFR+fpAACmSqEYgyT47GKEghz81LcfXyk2tZ12fqpYeXcdnO5mQ7XM4Bm6Ke+HYPhLKPzGElB5fipbc8RS8PWLaIoHYCQEU2iRLqNQXII+oIyDSzFZvedld3kc3VZRqKJAMcrp4MmY4qDolL2aE1pyU43kKa0aH249ihlZIfcvZiLLv6DRL5YCiIjzCi1TJncyOGZlChFj8huBDMpmBEwZGZI6tjCUCdYIVIIIaBlxQJT1vmgQ3FZg9DXbT1t0Vd3DpNkZhIsy3Nwg3d4qCpTUQeAzEylkQNDG81GYzA3s2K2c9uVsitlru5mPQbckdkFbpK0YTckDSV6AJmZPcM2kOpUNosIdxY3wjclkK1Bjh64HJfxZcwOAIJgUA7IAhUjJJAAIkLDxkkVMyQye2/Da6aW2deYK2sdOAQiW6hpg1ttlktXbZE4n/D8tC3LepjmYtOWvfd0s9KXTrCAs2nvTBVHzJaVRBowXgkLzc0dTjhYploOk0/Oyc2p3qJ1JpjshstESFJCRDBF0kgjzIVAS+SG4kB2g0gvRguE2RBO10SKFIx0864EkKlgbq2ZkKYWQVqBGVBgBFpkZjpZ66B6lDlgGQn0CNHdYExlkpMsU0HrgKnttzPa6sdUr/b8xGXR7kphOWQUAGU59d4ilQZM7mBOYmUCkVRmGlWI6lbMLYtSzpycUykGhaRESmMUlEWXGw0gkYPpVAQGbWUshQlIYgoUDdXLZEYlKWP2GMwVJBaiGgEiBSQAwHoIUqYi1dEBufnOiyEzUmlWnCSoRBdAysBUOkSTmbsxMiObeoPDzF3pPbbztoJrZs719MRtI0RDmmloTuXpeT0/LcuyRA9S7piTDusDfI5ORJByg9OQKlSFqOwxpDy6mZkgQVCij/fFdCZpPRKQO0SOKyzRSCfcWYsbFFvvMf6+lMZcDnO4EUIAgmTS+A1gPPHIDMhlziwIKKFLd4RpMK6GTCUyCBnTDG7OYtl6Kkb5x9a1tm5blBKZbLmsap0YU60NMlbl46fT8dPx8em4to4UkSbjGMAQIUHqsiKBcoeLxcxh0WPbskXCMNdanRRtGCozUilEOmiKBJDgwEM0GkgzFmMxGjIi1r623tulzQuQOc1GX5SniYoXHHoZISPswnQoIt1hZmNaGu1vaHk9MyMyM3GZs1RoNmBsZFpEh5q2LUoZFTeHlNJ6ZinFcAGlKL98PJ0ejsenc28xjZHggoYv+ICgpBQTNGa5QBVmqHdsAZnMZW5uYMAhDbQQEsgy+n4CIO2fIw3IBKlUtIgNsWVEmtFocpM7ivuYEsrAIzAwxzNk5niouLBBw4JL5T+gvBmLmAQismUkiNYNVs1rsfEuI3tCTMPWsm5yZ3FFtG3btpY5kxw3hmT59LguT+u6bNazDMQ69MILMTxUOWVmhDovFGLi8lskU2iZlkzSQXeIitCaGSQyC0aFHdM7AJhRQs9QRCgbIoAgkmbmxiyO4uOPmSkvI6FRCTcjOcgxuJkQGjOouTEdkEnEBR5hvPieGaJIolU3YwUACkrIAan37IEMk/cc/xeSCNqAykB5PG193Xpr1rOA4WoiMjpTlAtGszQEQ9okKDt7JopVc1WCqYhYU07WigoC0bKfe8twkXJO5u4gbBQdgjTrPdeuFj0IGSAzGmgCiKFjUC9zmYNenMoxBxHpNmYYiwFnxlMsQBpgmUyKeXFMhJC4sBMZUgkz1FIiI8dp7JHbyl2l1UuDors5AHDUJZV1i+ijFioxKh4yozNpnMRqxWiWbvAIX2GEImOeuDO6wZJNULc0NMmQUCTVjTkyBbDCcW8AcdwRAhJD7DAAw6Sfuvw+5AiDQ4rBQBitmBkQEYAyBcA5OPIxrgNIo8MI2XD6D3KWlyrHl4EaAkwwYtCogJAXCdeANJg5x3/AJVcBlKV39F4lWoK2CZE53hBlNoZeOoxiXP6dREOou09WIYcMw+aRiWg0kJGsYpMgRSiQpTjBkGWimKeJUxaDmgFyG1N2agiykNCEWUKEiBzvOJRbRrwcwlG0nSkw/9lr7UIgMgkmYJSNonmpcxViIntktg6yW3d66Wyjo5EwlkFiEZFIQmJRdCqMKoNZSkmwoR8P4gswkcbhWTBo1KmeaJlUFlpxa6GlRWtddJKUJg7aYYxsYwa6KEx085JOBqnky1iQCEpGOm085AvbMPwMUvbINh49zUfvAQzKzLTCC3FcoGzR++h/2al0G9J8XgiEAQszMyLcBRDJiBgZgczUIDMEkGYje1UqRGg2TkmXXHCwmI2LciGBCb9gPVqagD66QI8w1mJeSmSEtG2ZUC1mJiuy8CQCGeSGrGNsIt3TDaDDEAbSR32GpWSDK+w9MlJgKe4GZWQOmpUSgkmMoUCAjQsm0uHVqzJ6j8ymiME+FPeAujoNiYikORPosJ6kDakcCkVX9lBKNlQSuVvPnpmlCgZMtOoqCaNVopqJAztAAIzFfapWzBXoLZWJ8aFesJOBCSCDgZwMdSKNATZkh5pCghKzmzvJHP0duDwndyNTIz5kBRDgmWH0UnyePHpv25ZJaVQak4EwDtB8eVZ0s7lY68pExjhAo2ih2As9rOhJAzvYYS1kTJiq0cGeIgbEc5E5PAYSgNHWzalClDHZkIUU1JVdSiPMitu+lKmUDC3q0STSvRpz9GZRNLp7AYtbNRA9QzlsjDRBZkKFlQvBIDFlCRgg0MyQmane+3jug2EeApcbFGEUFW5ezMbTuSAtAw276nOxudpg0HpXIgtBEEIxE8yATPUMuLVkKHsKEWQUn50oUFLuZiYDUtpah7m7lw6C1pGVDs9iqLwoo6O2GAuNbixkoeBU8UhCJAjzVGyt9Qgyi3Oi1+pOqUeL3HrGhTyzrA66m422NybATGXAjGYFJkCtKYZ0ZRAUMUY+urFWKAhjMWNKGgJZuFudymFXD7syz0iyVnozqDpN0Xp0KeTDhYkQNGJxwKBTmCqUT6bJNpdX0JRI8J/mg9JFig2oGscM5o4YqUwyR1CFgEzJEOnD/dSCl1CsxnDUQwmjeXEaMzLQu1oIw5AmhywvHNWYsV2XBCpfKJ0CZO85VHuQUEb03lhLGVDAfZD5MjOFhlXG3abqu7lMBWYqjuJWvECgVLyA3NrF/EQxh9JYVN1ttFF4LT5PFdW6oRS3wohuYHFfembPkrxYWwOmIX8Pg9BLlCdlEruwZiaKmwWU6ojeZYLLjEKM4RwjKasYn56bgaK7lYlWmaZtW0qKjjGeEaZiWau7CxQyzaMmJLnG2IRlZXQpLEkbnlzBoD5G6/RIZFpPtAjKoqln7+w0FislaEEGMiOE/uLqsWKluLH0HmDuJ7B0cqq02XPisPsiCVBeWBIOc0ij6oQYGQ45bODXnjRYiGsgL28AfSgomQoAongJdwopNaRRoDtRCNDnYrtiY1LqPUJMw5CxaCjFajV36z0Aluq1WPSeXT2sjU4YYjKH9G8cZFW8hEiVshbcWkSGldajtd4zihe6OekwFlO2NZQwM6vOaSrTVJ2+Lg2MydHBCyS9zMAel6neinmRzeZGrVAk0RimKEPxA5wsxUiD2JURck8QXdbhKTEUl6uki1SKQKoa3ayGJJnhULmbaOCyITMD7DHcVe6iigHMZO8msLiVYo25RQjMQYvKMDRrG51gTHxskX14Uq2FaSvZPTKy9chQMDMdZqUaSEaC0RNmqM7Jfa7FQaveRYltZZskN0yFdeAv0MyUAEqpc7FigGUTsyMLAHdjQao4vRTIW2iAD2YHmULCgEwMZSAJgGZkYkB5GwSCM0cRqZ4IKSIC6R7KUHoqaXSzlUT27gB8HjxFtGgt1DMBK4ZLzJsCEuILtazIBJmtd6SnuifEbSAm5GSEeSkO0tjbaMPQP0DpZCZH61y2eDquZ6tz2fnO6cyh9WpwZ1H2M+eY3OSJjBZhjWhmgNPSyOJmpMQICRGZg4jgZTS9CAIE3GDFgYvjJQdVZxLQIrgKmX0U1yGLhkIhE5spaaQCQJKZoRa9R7YWm+DOYZqzhAglaaUUHySVLDoJZO/RQ+FBsAUi4aGLRDq8Wy9uiZCyZ0NLBKeaiTV5ynhSa/s2TZp2nIZZCIyIwbKUt3dTnClGDvmyF2UzoFlCCAit7dzdrFYoDZFIuRMJpQ1bVo8QURwAhoIgSmYtI0nBc8tFnak0DxRKdiHMBDAild3MGQmqdx/WwUg1qWeKAKxguEZhRCluTuudGI4ShJQtIrI7AIug4EPcyYwIRvQWY+Tl1qFISN3GDODn9KeOkwWLpkPd78vkiRQ4rNLMyPL1F/fHh3Z+svUcKTCyZzv2bhTBMFBwafLiNkZ7wuCApME38TL5MSRGQBQoJsGePekSO8CeVFgdYn7WQRiYm/lgMCE4acZL/UvLtMwuKRLGzIFe8QI2xOKW8h5GAqmQMiEomDGqhBDB1tOQEb2nRAMtGREZYvZkBlnO4tKtTdrP5XAzHw6TQ0glrSNgNLL87lv/+Uc8mp+I1XzJtWvuW0OXgzIWYKoYhgUj5ISUIVGuNBKuQddjsNGyBEIKCeYD2yk1KHQXDFmQoukyI4+BKt1YipeC6shkHz4To+eFiBEFNsEz2RVSNw5dzy8ojgS7D4kjQwRkXb6EhTIzeiRoo540i63bNnYNOFepJznZ/m7/5t3+1ZUbEJmBbkPiFMo3XwRbeNCzGnYtoK4wZG8hGLDB42KtiBAjx2T9z6rlxqn4KA0Gk9Qzm9JoO6uEcMEXA2QCSCpeWD3TBUbTjKVwmqobWwsp+UKX60IqwdwAa5HIyJRfSj6Qo5LBfTDWl/UdEroMMaaAbAqnZWQoevaWPgQBIDdlmJXdfPfm6ssvDzc3FYwYFo7hhiHKq/223gc2ZzhytzauGyIvVtAu67Cuy+vryR4Jqbgj2S51nbUUCMwk2bK3yIhIfwHMSMDMSMhxmdQQcYH1oBlsMHEaUmP23iO7QZUuV+TwmZjBM9H7MLbLLav7ONFjii0jYYc0IhKhi0mqk1AmM5Up9CGbDRO5O1ILIg27w/zq7fz27bQ7GAaVQA11qhQrVeurA/FqQq/qsbVsGQGBPTPUrQFLBymOXtZjMK0QQjCI0kRzNxoj043uDEFQ9GbDjzUsIJfRd5zN8dQG7WxOStkjcg0oo4eSxYdfZlAUMiE6UhfPxVioIoS7Y3hTLz0ze9KBlhcOFCkbB8jUYogxEhCDcPWS4plpM+v9dPfOru+yluxJmEXEgOJKFETsS9GVtRVtU0vbsmw9hIJB9PRc+igMwz0sAj1BEHDg4skwG5QxDVatoAyQkngRKkZPk4MvX0ODX24nSWQiWpjjQi+7jSvVdbm9vMCvSOX4mkGf0C7n0zjGe0Aaw4Mu6m8O7j9SI1QtXIYM2lgyULr7/mZ/8+V0/0U97Ac+uShHIzbFRNlaq4X7ab2/NcBgtSd7sq4kI9fIU+9rnkUnJ8oLM7On3KzQ7MXqKkKGwZf7iLqIkUazzMTlV0Zg5WI8Lpd6o0GpDxpvNFk3G9zeiK0IF4ZJl6Ez6Y5heYJG1ocvV5pgjyxm6SItsydgzohsqUgV0sxSMDOZB9hB7Obrt3dffHf7+svDVCwDDREJJpxmCSjLum2KNI/dVO9vaubUoobstBV3cWvrYz8/tWj9YjwZlQU5ypGRbhc2I6WUZfahSkqQuTIjB6lNcuQwL8Lh5W8aUxJscFrDLPMP5SwFMWkwk/lFjTQITDMjnMg6CHEIGMdfZmaA28WO6STApJBmlm4kPSUYk0yrmPc3b26++O3rb76/fffqqniP3gevCWgEXSCU7GoZLnO33YzX9zVVBHveSileFcfd+oGn5Yjow+UxePUhayGR1EVrFBDJnpdCTaU4+IC8OMI5NEW+sOcp8bI3BhzE+HiIPTR0aQLD8gcT7KLPCDLIiVJZrJYRY8kBcg28IEmHUjlc3/EPvwRBRzH2LAlscNV5uju8/s3Nr393+PrL3aGaSVumweEmxihHAEpEDBRNmBsOu3jzqoicT+5lmlzF/bREi+jL0FlggNFEBJEpiQ6JmWLv6BKYThS7RAcHmztqSgQTsGFQHq/9kjdQ5oUjN0teTESjkYxHO+R2dalfPgyYW/VShIwMoGM0gxxXzHV5ASMPFJFKqSiNogW5ofQyz/eHN9/cff+HV7//7u7N9VQgJUGTnLg4rQgYWXrvpbD3AKyQXtarK2Z6KRWEzNal1nlW7TmWzTnQU5FpGI6gIYVG9MiMZIyA21gEFBdtGBcP8iWDOizLOZxfo5kN3TiSTod81DUO3974opQUySZ0gUoSdeyJgQkaZyehiOgjFGFWhNRACcrMDuR44/Cgqezn+9s3395++++v//D7++++3O+rFD2MKU/RhtZktBzhyYgBT0eVtIxadXM1u3tkLN0JJkycMdFRi2Q90PPyFiI1xsPUMM9oOG4RlwoiDBYnIgkKg0ylk1Kk5JcoIlIZuqRXB8RIXBToFAZOIC7HZexm6r0zImADlQgIqWduI50H+JDw0ROQsSc6zOVwl5f5+uruN/ff/dvr3//b/a+/PtxewaCeIViKTisIB3y084hyEbukjAx2Gs3bPNdSIjN4hA9+14uXYQzJCSyh1rJny965BbdkDvdyECPi4EabiFBf8vKwhinGzWupbtnbSKAMapgAfXAWvPAakcPDDB8JAgnSMCBK4yxaIIfLHcVAS6m96DEXf/RIE1uKBngAgGfxcthdfXn45vf3f/i3+2+/uXp9W0vBCxhHAWf3YsO5kn3d2nkt/7BsXFj0ZqmoudZJ05QtyvW+3N7WVTz2NHr1cig+gW3Npa2Z3bbAlr2V3nq0LQLqcpmrlBKDD9gUAgWAyQGfcNHCEsoM0szNxw5He0klJIddqdRaiklStoR8CG8KgCIxCKpMcMykGLmGFGSS0KEwJErmlMpad7vD4fDF/Zs/vP3Dv7/+/tubu1uUgqAPd5jR5+IzlL31tm3n5fj0fH4+FcMwYACIDAOZVFh49uK2q3x1b0sA7uVYW9fkNpdaRMbWs6SzTJMDbVtis76Uvra2rNFjiSgeY48lMxlkWhpTjIiuPijWUZIgubmV4V3rEUrYMAMSJjJH/3RCRAgmFo5r6u7DnRzsoR5UAyOHKRnu6KaU9eRmKeN02F1/9erd9+9+87vrb349v3o9l7krA+wEi7lD7OvW23o+L6dlfT6enp5Pj8/lcqaVpA0mLzMyGJ1hNOr21sJqme3wOB1PgFhQGEDkpOyQF6uuaV+hyla25/X54en0fG5SF13My5kaOToAiovGyuED6ZnDe+aC4RKTABBMpDLVU4JwwWgvGGwgC1zsMV2ZgRxaxMirZyqUUJeafAt0M6vTfH/76ps3v/rd62+/v/nii+vdvoy53EyCm4TY2nZejsenh6fz03E7nWPd+rL94xpiZEulyz8TEVuDobvn9WH2Uq52djzXtXlsXJcEfS6lRJqjFDinqfpUanuerKhLyxqr+hA48jIywEGMLPULoHgpmgmXjwV0pUSqhTouzzkiM0CJzARyLO0ZUScggczL0Dd8QjkC4UQqMxmqHaUJ3apPO90dDl9ev/317u1Xh92hSNlSCNLcnGitrev56enp4eH0cFyfT33d2BORxcwuaRy9PLRBOUdKcDdprRZlqodarw9cGs7nfD6iTOytohkko2DmFfMVdvt9Rx63WJ+2bTVeJjkUDycKrL/E5S5T4cD2Nn6G4la99FRkQEnS3DOQ+Q/ZlZLLIJS4DH6IRBMgTChmpSJRaApTjtB7p8sNu9luD+XN/vDl7v7LcnVnsGwRmWBSSeTWl/P5+fnp0+fj42M/bVo2tj6WlowgoEuZeYmCXPx8GUoZPJDmrVDVUPbazWUqXoqvzdpW2QqaonFNbbFZxm6a6s003ew9bRVz7RGaTFMxd3iia7DOHJ75YXQwHwln0Qxmmdoyuuju7pUmXtiOkZUemoANqTiADqQZxPQiL6WoVAupK5dOijLzWna3++svbt795vrt17vb16VO6KEBykkqIrbzcnw+fno4fn5opwVbsIdnju0Kxa2S1LAKUvDRWAa+E9XBKLCNsFSprXix3TTX0rIsK2IzLd4Wa0tsq/qJUXtX+qHsZElu3nCm0UtNd6AJXbLRqC5eWZLuL3Ao1dTPTc89JJpVs2qWgwaN9BE1zjSlDRd8CmM3s6B0U7Uy2eQ1iRbUlp2cJ+f1NL2++vK7uz/8693XX827XU1cFkab6A5F621djsfl+BznjVtnz4yeI7AoFHe/OFAgGeAXtmfYOIlVUqRzKM1dZt0Zpc47Zp2ZrUT1rc6b86SydT/2ZVuXlsmp7O52dZ5yt6/RCjZE64igyy6uL6ITigiaMpXkJvauc8tzV0UtblnNmVbMnWiWoMuQpLmX4BDkLi49efVpPx2upl2lpDVdW8I439T9u9tX39z89rd333+/e/ummnHrGQG+NOy+rdv59PT4sB5PaIGeyg52G9OvqdRiEDIAYxrzgpwxFL4+2IwEEGQm0lXqZMICYC4HVcW0WYurvS3V87TbFq6ys9oSWXaH6Vo7bTVmLnj++HnbVpnBtq7TVC1RWsKHZccI8bzlsuWWVKksntWTLGU2o4jC0g3mzS3mChKZlj3hrDcsU5mrXc08HHZWprVT5+g1sd/dvS3f/OHq23+9+/qLu9f3k3l0jlrFYmVKaDn103p+OvfnJU8reqCHsg+nHAVABU4C5hyGkSGeXOb/S1D/xektMREeoZUwmnnpLFacparsp7qf5sfpdITRlrRoK61dv9r/6vX1lSGe159/yJ///vz08bxuDNQUDoO5EjIVYIq9W08T6WUqB86Hej1P80QrBBSCbZi1KxW7KojrptqEiYdXu8PtfJi8GBM8NRyf88msX8fdV1e//t3t9/+6//Wvb6/2s1v03gPDh8ViVoil99Pz8enpcV3O6N0ijRSdiksoACgj5uZGhTLHOhxChkwzv4zxGgz7SD5ntDRzeolci6y6WeHBay2lepz2Pk11hT031t1297p+/d3NmyvHst2/rbYv54ynyN5J5DwskRc4BbB4YTWYqe7rzevp1ev9q+t5ngSnhDW6n3qk11omh0K2BLfOya5f7W/ud7vJI+3pHE/r+lnarur9u/k3/3r7+3959dWv7HpvZO997cqUXyybFCLb+Xx6elpPp95aGV5VaaS9dNFiWOB8mcVEogCAKRFgxiA89NIubES2UySJQM+NMq9mLneWWWaxm2fY9NT2z93ng9/e7e5fzW/v62y7u/uZpRzXdtwUp8DYMUAnwGEH81LNJzJMdV+v3sy3X1zf3U1TSRhaoB/XSPQwlolORAIyb3J0sw2icOr4uMbnnnmor746/Pb3+3/5w/3XX13Nu23w8p09B3cguJsze1tOz0+np+e+rOjBHK0npFHah2KAYj6o3RyxbIxNVRxmOmHwxf9orhrTu70ECTJgSUuDnF41z53kfitXe399v9td1Ve3+3nyabKrXd2Vui344e+nnz5syXXPUkEnCzjcyF5QpppeG4KV3BXuCmcfYvfa4+Hsn09onbWgFlQzN8/iHfG45IpWq56bPW4NO//i65tv/+X6998fvn63nyfvoZ4txhWyUbXlBJXr8nx6flpOZ23NIg1EXmzgfBHsAJTihRCyk4LzwgJc/gx9nDuwj2gEBGjsdBqh6FC2oFqKTYXToRbPwj5b3u6nV28Ob14d5hoEmAYl6TbvfDeXLff7eiARaUIdDl9LsMiqenTKwpaWyybJts0+PfQPn/HwxBY067Xg6lD2s1nhumVuuYS8ciGwr198ffWHf3/12+/3b16xerRY21APDZGQspAuY2aPdTk+L+dT9rCXWj50DnuRSS6eUkWOjMTYjCIgIzMvmQdC5ibIX0TRHHuIBCNIz5HWSsscrnSV4qWoWFSLqz3v76brfZ8cbWmPz+vH5/a8akWJOtl+nqr1ZUUki8M9glvX0ltL2jxNcKimaoavi56fcDxH656oI54nGoqBoowQp1J3++kwX72t33x/+P5312/fWrG1t9jkzXJSQbFUZheSpbihr+v5fD711oZ92igbLoCXAO5Q8AwsgDKQaUhIij4CAzFo8cju5uYau02UcCdTY5dEGpOGDHhBlr4Kdtzvy/6wn2Y/dWsd0eSzHU/LtuHpqF8+bA8fWjsOPSfDsVGg76YDqCZtGb03ODllqTSzCN/Sl2XZQmKZppGa0jSX/e56t6ssUcF59no1zTfl9r68+3L3xde7uzsWRYZ1qmfPxGZSBuETURTDvZRNcdpya9MI5iFiGJZ7vHj/S2SmsUToYlW4UAGXOMhwE40xKIERwNQQswmWEb8cDz9T3USDGyi1aZqvruy4ntcFn37xOJb1vJ3O/bTF+4fTw2k5tY3sLWuk9bTo8oypWqmhDJFZ3Konp0DtqdE861RvbDdKLxi7fb2/t8PB6uRlLoebeX897W787n6+f1MP14aSPSI01h0N2I3U2NwxzGSI1rdlacumSPuHweAy2WuoJ2PPAIQy7CvR+4UdH5rSSMENTfNyAXExnOsSR8dY0gBZobkSzcSUZ2/TPu5vy9qWhzP/9gN+tv2ybMu2tR4/f1o/H/uphVsunTuvPXJpSy7blde5ZJE6rKf1je5cG93kCJ/sZpqvc9SD8BJXt9PrN+Xqus67UiZMO5/3dXdV9td1d7BhATaNESoxBFizCy/8Er7rbVvPp62tiBih+WH9H3kd8KKUDJN+GScnUy85gFHb/vmwErIXz91l18jISb3kRd3GwxIZ2U3szO1qtte37dj2v3zYjkc9PW9LW6V4ftqejrk1Gv20cAZa57lpU1MZpuYSydasd5QSiTDDPGmarU61sAOAYZrs5s7uX5fDVZnmCpMQ5q0WFEnNI6Dh08nh/bh89w5ezsNoXFBEtKYx/aVMl3TxkCQgjsU9Q+ssL9nesZZUusCwfLmGkiSXD8gwVqpeRqJxODmEJismpbqS2ZfFq65n3F3p6am//7B+eGzH86qIvsa2CWKGrad4zhaR501qGR5KryytM3uwYCq5n/P2mtdXuDpgmlV8My+kpgn7Q+6veplUSiYVEdG0dY+o3ovVymIotIvrjyJ79EveYlAs4/32QMqGceQFKwwjGi72H+Ow1wp98Fd4+XIpI7rGTZRohsh+YaBImgw5ctNZMCLQUiGQ3a12JWLNKtbyZvccd35+ztO59a1trZGqRQZoU7Z2pnpq2RTKxhDiuhYDi3N38Ne39ubW397j7jr2++Y16OmWJMzSS/Tt3NqIuxmgEVAofbZWfJ7rPLuqFXdmpGhwDP3SL4cCGb333pjBS7G5XB9CgXwRT8iEEQUXCWpsd7o865cCl8Q/77glOXTxYYMEeSErBAyft1iyxaCBzc32ePri+tC+qMWmz4f6+LQ8r21do63oyyiIvXe0ri7klrtd5o6HyQ+1vHp19e6df/Hab6/b9a5Z2UCZV43BIjJ7btF6hnuhFwyDVvWeay7gVKfd1X5/Ne13NhWaOKgOYKQx3QuBjOhb6z0uKSTBhv6o/EcF4su9LGoxqtWw7jpo0IUkiguJO643DElFdI59VjG81j4MHdYBKmOlcwMu5dCnyfGru7ya8vgmz6s/ne341J+OenrWtmlb8+lkkYXIsuO0qzc38+trvL7K16/ji/vt9gZeuteI3jJT6zJY5pFxTRKG7A3slxxHayNYaKWt56ZtU9/X/Vx2O2dB1g4FAoRbMLOnthiKSRoClkqZgzlORQIJo9yDo8BfqOyhEojkP2nTF7J5DNOjXQ7CDsTFEMahRg3n4zjGBCOk2bpNMbO+3vmrvXd4a+V4yufneHjcHk/5+clnj6s907m78tf39YtXfHut+9u8uXq+2pd58lTrrW/rGpkORES8eLNlJoluAGF0M9hY2EgrgdYTSWX0qD2n3ezVSDhFsnqB2qhol2Tk8JUNC8YIyODlVJEkizKJfyTWqbFvdSCuMQ5e9BRBQvBC1l+cHC/RlcufG/tYBwWWlDYFOsDmXmqZaI5Ztztfr/x4XT4/x/udXc0RQt2X21fz67t6e8DtTtdXKON7gMmir9l625okmPXWRs8eTTl7WiljKSPNR5u2EYNJxQkNppQi0LPuyGKoZnQXCfPhpMHLriRerpFeTgeHoQlMoIxhcTAx4zGOmnVRL/hymv6RLf7/+6GXrOcFcBDIuIjxYz0Kp+FbL+yUudJN01RL9Wmyacd5tvuroHPe+/WtXx9iLn1fbZ4YaVtu0dVby5bqHRpLwSUo4xIEzniJ2kFyaTgZLpoZM7NJ0Xqsq1pkD5+mepgdTo7N/Bzf3YCwjDC82JcuKGPck5dUGDFyEkPDvMDXf169scF//Jx4cdBe4pDDAoDhfsoEYDRkxjCsuwmIsdxEhOSZju5OI0rljdlc7f6q0FSqii/OXkXLKTbbQoHoPWINRSqboAyNTXkY49KgMQe1S7ywmC+mQ3MGMxKRbTO16K1N+71xTGuQEFtyuMeGU8eQIUB0mg2EoUtA1a2MKzc+nPGG/4EzhiXTzEa/u8hVY8Lh5bG/RDuG9/GfNQ4advdm4khfZo7WObwLImjkbsJcBqxLxRY9Zd6sgWySGbJHjH082YBEFllCGsu0MwNDiOKwXYp0GZRKyMSRO7UY30TllJdwq9DFeZcxFqsZ4cMKNm7doPaMI5X2gsPJMrDCP2/WpVT9Y0LkC/i6lPXLdRxWWl7uOS6rYgWkmV/WMgz7nTR4ME+CTKVlBpskZY4SmhkGxfg+AsP2Z4BbBrNnxIuKqTRmtBwrkzIv9slxBYzMVPSg29jwMXb9xFgmWMxgioh1ayBTNqK0vb8kEIavhDZiTQOZp0yDQjcA/ycuz17r5zo9KQAAAABJRU5ErkJggg==
"/>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>(100, 100, 3)
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=795d44b9">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="Create-Labels-for-Negative-(normal)--and-Positive-(malignant):">Create Labels for Negative (normal)  and Positive (malignant):<a class="anchor-link" href="#Create-Labels-for-Negative-(normal)--and-Positive-(malignant):">¶</a></h4><p>Labels will be alternated to match instances</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=e8000e18">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [98]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">neg_labels</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">*</span><span class="mi">224</span>  
<span class="n">pos_labels</span> <span class="o">=</span> <span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">*</span><span class="mi">224</span>
<span class="n">labels</span><span class="o">=</span><span class="p">[]</span>
<span class="k">for</span> <span class="n">n</span><span class="p">,</span><span class="n">p</span> <span class="ow">in</span> <span class="nb">zip</span> <span class="p">(</span><span class="n">neg_labels</span><span class="p">,</span> <span class="n">pos_labels</span><span class="p">):</span>
    <span class="n">labels</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">n</span><span class="p">)</span>
    <span class="n">labels</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">p</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=6f86e703">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Train/Test-Split">Train/Test Split<a class="anchor-link" href="#Train/Test-Split">¶</a></h3><p>First, negative and positive instances are evenly distburshed and then 75% of instances with labels are reserved for the training data.</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=891c073b">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [92]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">data</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">n</span><span class="p">,</span><span class="n">p</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">neg_arrays</span><span class="p">,</span><span class="n">pos_arrays</span><span class="p">):</span>
    <span class="n">data</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">n</span><span class="p">)</span>
    <span class="n">data</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">p</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">'75</span><span class="si">% o</span><span class="s1">f the data is :'</span><span class="p">,</span><span class="nb">len</span><span class="p">(</span><span class="n">data</span><span class="p">)</span><span class="o">*</span><span class="mf">.75</span><span class="p">)</span>
<span class="n">X_train</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">data</span><span class="p">[:</span><span class="mi">336</span><span class="p">])</span>
<span class="n">X_test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="mi">336</span><span class="p">:])</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">'Length of entire data set: '</span><span class="p">,</span><span class="nb">len</span><span class="p">(</span><span class="n">X_train</span><span class="p">)</span><span class="o">+</span><span class="nb">len</span><span class="p">(</span><span class="n">X_test</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">X_train</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">X_test</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>75% of the data is : 336.0
Length of entire data set:  448
(336, 100, 100, 3)
(112, 100, 100, 3)
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=aca8af7c">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [99]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">Y_train</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">labels</span><span class="p">[:</span><span class="mi">336</span><span class="p">])</span>
<span class="n">Y_train</span> <span class="o">=</span> <span class="n">Y_train</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">336</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>
<span class="n">Y_test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">labels</span><span class="p">[</span><span class="mi">336</span><span class="p">:])</span>
<span class="n">Y_test</span> <span class="o">=</span> <span class="n">Y_test</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">112</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">'Length of labels: '</span><span class="p">,</span><span class="nb">len</span><span class="p">(</span><span class="n">Y_train</span><span class="p">)</span><span class="o">+</span><span class="nb">len</span><span class="p">(</span><span class="n">Y_test</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">Y_train</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">Y_test</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Length of labels:  448
(336, 1)
(112, 1)
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=902d067b">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h2 id="Creating-Model">Creating Model<a class="anchor-link" href="#Creating-Model">¶</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=23e1f1d8">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Import-Necessary-Libraries">Import Necessary Libraries<a class="anchor-link" href="#Import-Necessary-Libraries">¶</a></h3>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=60fcf547">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [87]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">tensorflow.keras.models</span> <span class="kn">import</span> <span class="n">Sequential</span>
<span class="kn">from</span> <span class="nn">tensorflow.keras.layers</span> <span class="kn">import</span> <span class="n">Conv2D</span><span class="p">,</span> <span class="n">MaxPooling2D</span><span class="p">,</span> <span class="n">Flatten</span><span class="p">,</span> <span class="n">Dense</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=84b2eb73">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Create-the-Model">Create the Model<a class="anchor-link" href="#Create-the-Model">¶</a></h3><p>Here a convolutional-pooling filtering is used that scans 3x3 submatrices within the feature map and chooses only the max value to be represented in a much smaller 2x2 matrix per each 3x3 matrix that was scanned. The aforementioned filtering drastically reduces computational constraints and even though this data set is very small it has been implemented purely for heuristic reasons. Note, the use of the relu activation function creates an ouput of 0 or 1 for the presence of each feature reducing the uncertainty if intermediary values between 0 and 1 were used. Also, the input shape must match the input of the 100 x 100 x 3 image files for the first network layer. Finally, a sigmoidal curve is used to output are binary class that will determine if the image represents a lymphocyte that is normal (negative) or malignant (positive).</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=ef3007ed">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [88]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># input for a 100x100 RBG image</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">Sequential</span> <span class="p">([</span>
    <span class="n">Conv2D</span><span class="p">(</span><span class="mi">32</span><span class="p">,</span> <span class="p">(</span><span class="mi">3</span><span class="p">,</span><span class="mi">3</span><span class="p">),</span> <span class="n">activation</span><span class="o">=</span><span class="s1">'relu'</span><span class="p">,</span> 
    <span class="n">input_shape</span> <span class="o">=</span> <span class="p">(</span><span class="mi">100</span><span class="p">,</span><span class="mi">100</span><span class="p">,</span><span class="mi">3</span><span class="p">)),</span> 
    <span class="n">MaxPooling2D</span><span class="p">((</span><span class="mi">2</span><span class="p">,</span><span class="mi">2</span><span class="p">)),</span>

    <span class="n">Conv2D</span><span class="p">(</span><span class="mi">32</span><span class="p">,</span> <span class="p">(</span><span class="mi">3</span><span class="p">,</span><span class="mi">3</span><span class="p">),</span> <span class="n">activation</span><span class="o">=</span><span class="s1">'relu'</span><span class="p">),</span> 
    <span class="n">MaxPooling2D</span><span class="p">((</span><span class="mi">2</span><span class="p">,</span><span class="mi">2</span><span class="p">)),</span>

    <span class="n">Flatten</span><span class="p">(),</span>
    <span class="n">Dense</span><span class="p">(</span><span class="mi">64</span><span class="p">,</span> <span class="n">activation</span> <span class="o">=</span> <span class="s1">'relu'</span><span class="p">),</span>
    <span class="n">Dense</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">activation</span> <span class="o">=</span> <span class="s1">'sigmoid'</span><span class="p">)</span>
<span class="p">])</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=6d909324">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Compilation-of-the-model">Compilation of the model<a class="anchor-link" href="#Compilation-of-the-model">¶</a></h3><p>Compilation of the model incorporates the loss function into the model architecture to determine the optimal model</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=96bc6d78">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [89]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># compile mode</span>
<span class="c1"># categorical output = loss and metrics</span>
<span class="n">model</span><span class="o">.</span><span class="n">compile</span><span class="p">(</span><span class="n">loss</span> <span class="o">=</span><span class="s1">'binary_crossentropy'</span><span class="p">,</span>
    <span class="n">optimizer</span><span class="o">=</span><span class="s1">'adam'</span><span class="p">,</span> <span class="n">metrics</span><span class="o">=</span><span class="p">[</span><span class="s1">'accuracy'</span><span class="p">]</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=df0fb661">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Fit-Data-to-Model">Fit Data to Model<a class="anchor-link" href="#Fit-Data-to-Model">¶</a></h3>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=6f9a0557">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [101]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># fit training data to model</span>
<span class="c1"># rerun this cell to 'continue training' - don't recompile</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">Y_train</span><span class="p">,</span> <span class="n">epochs</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">64</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Epoch 1/5
6/6 [==============================] - 1s 209ms/step - loss: 0.1979 - accuracy: 0.9405
Epoch 2/5
6/6 [==============================] - 1s 205ms/step - loss: 0.1052 - accuracy: 0.9792
Epoch 3/5
6/6 [==============================] - 1s 202ms/step - loss: 0.0428 - accuracy: 0.9881
Epoch 4/5
6/6 [==============================] - 1s 208ms/step - loss: 0.0115 - accuracy: 1.0000
Epoch 5/5
6/6 [==============================] - 1s 202ms/step - loss: 0.0078 - accuracy: 1.0000
</pre>
</div>
</div>
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[101]:</div>
<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain" tabindex="0">
<pre>&lt;keras.callbacks.History at 0x20f87b43eb0&gt;</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=f60097e9">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Evaluate-the-Model">Evaluate the Model<a class="anchor-link" href="#Evaluate-the-Model">¶</a></h3><p>Initial accuracy of 100% indicates near perfect modelling; however, as one knows that we may be overfitting the data or just may need more data points. However, let's check the model performance:</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=1be678a8">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [102]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># evaluate performance</span>
<span class="n">model</span><span class="o">.</span><span class="n">evaluate</span><span class="p">(</span><span class="n">X_test</span><span class="p">,</span> <span class="n">Y_test</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>4/4 [==============================] - 0s 35ms/step - loss: 0.5580 - accuracy: 0.9018
</pre>
</div>
</div>
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[102]:</div>
<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain" tabindex="0">
<pre>[0.5579643249511719, 0.9017857313156128]</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=0cab5c45">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Make-predictions">Make predictions<a class="anchor-link" href="#Make-predictions">¶</a></h3><p>Note, the y_pred value that is the output of the sigmoidal curve consist of a probability value between 0 and 1 that the image represents the positive or negative label. In this case, a value greater than 0.5 is indicative that the image is classified as a malignant cell. To verbalize this notion the probability vlaue has been articulated with a conidition statement to print out 'malignant' or 'normal' alongside the associated picture.</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=e8695b07">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [117]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># make predictions</span>
<span class="k">def</span> <span class="nf">predict</span><span class="p">():</span>
    <span class="kn">import</span> <span class="nn">random</span>
    <span class="n">idx</span> <span class="o">=</span> <span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">Y_test</span><span class="p">))</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">X_test</span><span class="p">[</span><span class="n">idx</span><span class="p">,</span> <span class="p">:],</span> <span class="n">interpolation</span><span class="o">=</span><span class="s1">'nearest'</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
    <span class="n">y_pred</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">[</span><span class="n">idx</span><span class="p">,</span> <span class="p">:]</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">100</span><span class="p">,</span> <span class="mi">100</span><span class="p">,</span> <span class="mi">3</span><span class="p">))</span>
    <span class="k">if</span> <span class="n">y_pred</span> <span class="o">&gt;</span><span class="mf">0.5</span><span class="p">:</span>
        <span class="n">pred</span> <span class="o">=</span> <span class="s1">'malignant'</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">pred</span> <span class="o">=</span> <span class="s1">'normal'</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">pred</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=ca3c3256">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [118]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">predict</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="jp-needs-light-background" src="data:image/png;base64,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
"/>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>malignant
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=945a2823">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [119]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">predict</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="jp-needs-light-background" src="data:image/png;base64,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
"/>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>normal
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=1f147b11">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [120]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">predict</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="jp-needs-light-background" src="data:image/png;base64,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
"/>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>malignant
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=a8c5e171">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>Just for demonstration we see that three randomly picked cells are actually classified correctly.</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=906a10b9">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h2 id="Summary">Summary<a class="anchor-link" href="#Summary">¶</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=78887a20">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>The convoluted neural network classifier for malignant lymphocytes proved to be fairly successful. The accuracy in the testing phase proved to be 90% while training accuracy approached 100%. The model performs excellent but secondary testing should be performed on a larger data set as well as possibly added in more cell types. This will test if the model has actually overfit in the training phase where generalization error may be present if the system is used in the 'real-world"
<br/><br/>
My next image classifier would include a multilabel classifier that is able to differentiate among the five major white blood cells (eosinophils, basophils, lymphocytes, neutrophils, and monocytes). Overall, the model proves to be fairly promising though even with slight overfitting. There is much promise that cell classification could serve as a great service to development of educational software for laboratory professionals as well as potential commerical usage if data sets may be obtain.</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=f0dccdc6">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h2 id="References:">References:<a class="anchor-link" href="#References:">¶</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell" id="cell-id=6ed01b56">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<ul>
<li><a href="https://keras.io/">https://keras.io/</a></li>
<li><a href="https://pillow.readthedocs.io/en/stable/">https://pillow.readthedocs.io/en/stable/</a></li>
<li><a href="https://imagebank.hematology.org/">https://imagebank.hematology.org/</a></li>
</ul>
</div>
</div>
</div>
</div>
</main>
</body>
</html>
