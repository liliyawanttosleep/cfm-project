param([string]$Docx, [string]$Pdf)
# Polish with Word itself: page setup, heading styling, table borders and widths, page footer. Then export a PDF to check.
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$d = $w.Documents.Open($Docx)
try {
    $ps = $d.PageSetup
    $ps.TopMargin = [single]64; $ps.BottomMargin = [single]64; $ps.LeftMargin = [single]72; $ps.RightMargin = [single]72

    function St($i) { $v = $i; return $d.Styles.Item([ref]$v) }
    function SetStyle($name, $size, $bold, $color, $before, $after) {
        $s = St $name
        $s.Font.Name = 'Calibri'; $s.Font.Size = [single]$size; $s.Font.Bold = [int]$bold; $s.Font.Italic = 0; $s.Font.Color = [int]$color
        $s.ParagraphFormat.SpaceBefore = [single]$before; $s.ParagraphFormat.SpaceAfter = [single]$after
    }
    # Word colors are BGR
    $navy = 0x3D2114; $orange = 0x125CB3; $gray = 0x826B5B
    SetStyle -63 26 1 $navy 0 4
    SetStyle -75 12 0 $gray 0 14
    SetStyle -2 18 1 $navy 22 8
    SetStyle -3 13 1 $navy 16 4
    SetStyle -4 10.5 1 $orange 8 3
    foreach ($n in -1, -67, "First Paragraph", "Compact") {
        $s = St $n
        $s.Font.Name = 'Calibri'; $s.Font.Size = [single]11
    }
    foreach ($n in -67, 'First Paragraph') {
        $s = St $n
        $s.ParagraphFormat.SpaceBefore = [single]0; $s.ParagraphFormat.SpaceAfter = [single]7; $s.ParagraphFormat.LineSpacingRule = 5; $s.ParagraphFormat.LineSpacing = [single]13.8
    }
    $s = St -63; $s.ParagraphFormat.Alignment = 0
    $s = St -75; $s.ParagraphFormat.Alignment = 0

    # Each part starts on a new page, except the first
    $first = $true
    foreach ($p in $d.Paragraphs) {
        if ($p.OutlineLevel -eq 1) {
            if ($first) { $first = $false } elseif ($p.Range.Text -notmatch '^Speaking order') { $p.PageBreakBefore = -1 }
        }
    }

    $s = St 'Verbatim Char'; $s.Font.Name = 'Consolas'
    foreach ($m in $d.OMaths) { $m.Range.Paragraphs.Item(1).KeepWithNext = -1 }

    $grid = 0xE3DDD9
    foreach ($t in $d.Tables) {
        $t.AllowAutoFit = $true
        $t.AutoFitBehavior(2)   # fit to window width
        $t.Borders.Enable = 1
        $t.Borders.InsideColor = $grid; $t.Borders.OutsideColor = $grid
        $t.TopPadding = [single]2.5; $t.BottomPadding = [single]2.5; $t.LeftPadding = [single]5; $t.RightPadding = [single]5
        $t.Range.Font.Size = [single]10
        $t.Rows.AllowBreakAcrossPages = 0
        $t.Rows.Item(1).Shading.BackgroundPatternColor = 0xF3F0ED
        $t.Rows.Item(1).Range.Font.Bold = 1
        $t.Rows.Item(1).HeadingFormat = -1
        # keep a table on one page, and leave a gap before the text that follows it
        $t.Range.ParagraphFormat.KeepWithNext = -1
        $t.Rows.Item($t.Rows.Count).Range.ParagraphFormat.KeepWithNext = 0
        $next = $t.Range.Next(4, 1)
        if ($next -ne $null -and $next.ParagraphFormat.OutlineLevel -eq 10) { $next.ParagraphFormat.SpaceBefore = [single]8 }
    }

    # Page number in the footer
    $f = $d.Sections.Item(1).Footers.Item(1)
    $f.Range.Text = ('Flow Matching vs. Diffusion: Speech Script ' + [char]0x00B7 + ' Page ')
    $r = $f.Range; $r.Collapse(0)
    [void]$f.Range.Fields.Add($r, 33)
    $f.Range.Font.Size = [single]9; $f.Range.Font.Color = $gray; $f.Range.Font.Name = 'Calibri'

    $d.Save()
    $d.ExportAsFixedFormat($Pdf, 17)
    "pages: " + $d.ComputeStatistics(2)
    "equations: " + $d.OMaths.Count
    "tables: " + $d.Tables.Count
} finally {
    $d.Close(0)
    $w.Quit()
}
