<#
.SYNOPSIS
    Batch-runs WallMan "Edit -> Check Database" for WinProp ODB files.

.DESCRIPTION
    WallMan 2020 does not expose Check Database through the public WinProp API.
    This script calls WallMan's native menu command (ID 32958), then saves the
    repaired database through the standard Save As dialog.

    Safety and resume behavior:
      * source ODB files are never overwritten;
      * output is written beside the source as <name>_checked.odb;
      * existing _checked.odb files are skipped;
      * processing is sequential, so only one WallMan process is resident;
      * a temporary ODB is renamed to the final name only after a successful save;
      * rerunning the same command resumes the batch automatically.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\batch_check_winprop_odb.ps1 `
      -RootPath 'D:\桌面\dac\512mdata'

.EXAMPLE
    # Preview which files would be processed.
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\batch_check_winprop_odb.ps1 `
      -RootPath 'D:\桌面\dac\512mdata' -DryRun

.EXAMPLE
    # Process one source ODB as a smoke test.
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\batch_check_winprop_odb.ps1 `
      -RootPath 'D:\桌面\dac\512mdata' -Limit 1
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RootPath,

    [string]$WallManPath = 'C:\Program Files\Altair\2020\feko\bin\WallMan.exe',

    [ValidateRange(0, 2147483647)]
    [int]$Limit = 0,

    [ValidateRange(5, 600)]
    [int]$OpenTimeoutSec = 300,

    [ValidateRange(5, 3600)]
    [int]$CheckTimeoutSec = 300,

    [ValidateRange(5, 600)]
    [int]$SaveTimeoutSec = 20,

    [ValidateRange(1, 4)]
    [int]$SaveConcurrency = 1,

    [ValidateRange(1, 5)]
    [int]$MaxAttempts = 2,

    [ValidateRange(0, 60)]
    [int]$RetryDelaySec = 1,

    [ValidateRange(1, 16)]
    [int]$ShardCount = 1,

    [ValidateRange(0, 15)]
    [int]$ShardIndex = 0,

    [string]$SourceListPath = '',

    [string]$OutputRoot = '',

    [ValidatePattern('^_[A-Za-z0-9_-]+$')]
    [string]$OutputSuffix = '_checked',

    [string]$LogPath = '',

    [ValidateRange(0, 120)]
    [int]$DiagnosticPauseSec = 0,

    [switch]$DryRun,

    [switch]$StopOnError
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $RootPath -PathType Container)) {
    throw "RootPath is not a directory: $RootPath"
}
if (-not (Test-Path -LiteralPath $WallManPath -PathType Leaf)) {
    throw "WallMan.exe was not found: $WallManPath"
}
if ($ShardIndex -ge $ShardCount) {
    throw "ShardIndex must be in the range 0..$($ShardCount - 1) when ShardCount is $ShardCount."
}

$resolvedRoot = (Resolve-Path -LiteralPath $RootPath).Path
$resolvedWallMan = (Resolve-Path -LiteralPath $WallManPath).Path
$resolvedOutputRoot = ''
if (-not [string]::IsNullOrWhiteSpace($OutputRoot)) {
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    $resolvedOutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path
}
if (-not [string]::IsNullOrWhiteSpace($SourceListPath) -and
    -not (Test-Path -LiteralPath $SourceListPath -PathType Leaf)) {
    throw "SourceListPath is not a file: $SourceListPath"
}

Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public sealed class WallManWindowInfo
{
    public IntPtr Handle;
    public IntPtr Owner;
    public string ClassName;
    public string Text;
    public bool Visible;
    public bool Enabled;
    public int ControlId;
}

public static class WallManNative
{
    private const uint WM_COMMAND = 0x0111;
    private const uint WM_CLOSE = 0x0010;
    private const uint WM_SETTEXT = 0x000C;
    private const uint BM_CLICK = 0x00F5;
    private const uint GW_OWNER = 4;
    private const uint SMTO_BLOCK = 0x0001;
    private const uint SMTO_ABORTIFHUNG = 0x0002;

    private delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern bool EnumChildWindows(IntPtr parent, EnumWindowsProc callback, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int maxCount);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetClassName(IntPtr hWnd, StringBuilder className, int maxCount);

    [DllImport("user32.dll")]
    private static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern bool IsWindowEnabled(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern bool IsWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern IntPtr GetWindow(IntPtr hWnd, uint command);

    [DllImport("user32.dll")]
    private static extern IntPtr GetMenu(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern int GetDlgCtrlID(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern IntPtr GetDlgItem(IntPtr hDlg, int controlId);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool PostMessage(IntPtr hWnd, uint message, UIntPtr wParam, IntPtr lParam);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern IntPtr SendMessage(IntPtr hWnd, uint message, IntPtr wParam, string lParam);

    [DllImport("user32.dll")]
    private static extern IntPtr SendMessage(IntPtr hWnd, uint message, IntPtr wParam, IntPtr lParam);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern IntPtr SendMessage(IntPtr hWnd, uint message, IntPtr wParam, StringBuilder lParam);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern IntPtr SendMessageTimeout(
        IntPtr hWnd,
        uint message,
        UIntPtr wParam,
        IntPtr lParam,
        uint flags,
        uint timeoutMs,
        out UIntPtr result);

    [DllImport("user32.dll")]
    private static extern bool ShowWindow(IntPtr hWnd, int command);

    private static string ReadWindowText(IntPtr hWnd)
    {
        StringBuilder value = new StringBuilder(2048);
        GetWindowText(hWnd, value, value.Capacity);
        return value.ToString();
    }

    private static string ReadClassName(IntPtr hWnd)
    {
        StringBuilder value = new StringBuilder(512);
        GetClassName(hWnd, value, value.Capacity);
        return value.ToString();
    }

    private static WallManWindowInfo DescribeWindow(IntPtr hWnd)
    {
        return new WallManWindowInfo {
            Handle = hWnd,
            Owner = GetWindow(hWnd, GW_OWNER),
            ClassName = ReadClassName(hWnd),
            Text = ReadWindowText(hWnd),
            Visible = IsWindowVisible(hWnd),
            Enabled = IsWindowEnabled(hWnd),
            ControlId = GetDlgCtrlID(hWnd)
        };
    }

    public static WallManWindowInfo[] TopWindows(int processId)
    {
        List<WallManWindowInfo> windows = new List<WallManWindowInfo>();
        EnumWindows(delegate(IntPtr hWnd, IntPtr lParam) {
            uint ownerProcessId;
            GetWindowThreadProcessId(hWnd, out ownerProcessId);
            if (ownerProcessId == (uint)processId) {
                windows.Add(DescribeWindow(hWnd));
            }
            return true;
        }, IntPtr.Zero);
        return windows.ToArray();
    }

    public static WallManWindowInfo[] ChildWindows(IntPtr parent)
    {
        List<WallManWindowInfo> windows = new List<WallManWindowInfo>();
        EnumChildWindows(parent, delegate(IntPtr hWnd, IntPtr lParam) {
            windows.Add(DescribeWindow(hWnd));
            return true;
        }, IntPtr.Zero);
        return windows.ToArray();
    }

    public static bool HasMenu(IntPtr hWnd)
    {
        return GetMenu(hWnd) != IntPtr.Zero;
    }

    public static bool Exists(IntPtr hWnd)
    {
        return IsWindow(hWnd);
    }

    public static bool SendCommandSync(IntPtr hWnd, uint commandId, uint timeoutMs)
    {
        UIntPtr result;
        return SendMessageTimeout(
            hWnd,
            WM_COMMAND,
            new UIntPtr(commandId),
            IntPtr.Zero,
            SMTO_BLOCK | SMTO_ABORTIFHUNG,
            timeoutMs,
            out result) != IntPtr.Zero;
    }

    public static bool PostCommand(IntPtr hWnd, uint commandId)
    {
        return PostMessage(hWnd, WM_COMMAND, new UIntPtr(commandId), IntPtr.Zero);
    }

    public static bool PostClose(IntPtr hWnd)
    {
        return PostMessage(hWnd, WM_CLOSE, UIntPtr.Zero, IntPtr.Zero);
    }

    public static bool SetText(IntPtr hWnd, string text)
    {
        return SendMessage(hWnd, WM_SETTEXT, IntPtr.Zero, text) != IntPtr.Zero;
    }

    public static void Click(IntPtr hWnd)
    {
        SendMessage(hWnd, BM_CLICK, IntPtr.Zero, IntPtr.Zero);
    }

    public static int ProgressPosition(IntPtr hWnd)
    {
        // PBM_GETPOS = WM_USER + 8
        return SendMessage(hWnd, 0x0408, IntPtr.Zero, IntPtr.Zero).ToInt32();
    }

    public static string[] ListBoxItems(IntPtr hWnd)
    {
        // LB_GETCOUNT, LB_GETTEXTLEN, and LB_GETTEXT.
        int count = SendMessage(hWnd, 0x018B, IntPtr.Zero, IntPtr.Zero).ToInt32();
        if (count <= 0 || count > 100000) {
            return new string[0];
        }

        List<string> items = new List<string>();
        for (int index = 0; index < count; index++) {
            int length = SendMessage(hWnd, 0x018A, new IntPtr(index), IntPtr.Zero).ToInt32();
            if (length < 0) {
                continue;
            }
            StringBuilder value = new StringBuilder(length + 2);
            SendMessage(hWnd, 0x0189, new IntPtr(index), value);
            items.Add(value.ToString());
        }
        return items.ToArray();
    }

    public static IntPtr DialogItem(IntPtr dialog, int controlId)
    {
        return GetDlgItem(dialog, controlId);
    }

    public static void Hide(IntPtr hWnd)
    {
        ShowWindow(hWnd, 0);
    }
}
'@

function Get-MainWindow {
    param(
        [int]$WallManProcessId,
        [string]$SourceName
    )

    $windows = [WallManNative]::TopWindows($WallManProcessId)
    $afxWindows = @($windows | Where-Object {
        $_.ClassName.StartsWith('Afx:', [System.StringComparison]::OrdinalIgnoreCase) -and
        [WallManNative]::HasMenu($_.Handle)
    })

    foreach ($window in $afxWindows) {
        if ($window.Text.IndexOf($SourceName, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
            return $window.Handle
        }
    }

    return [IntPtr]::Zero
}

function Wait-MainWindow {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$SourceName,
        [int]$TimeoutSec
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $answeredDialogs = New-Object 'System.Collections.Generic.HashSet[string]'
    do {
        $Process.Refresh()
        if ($Process.HasExited) {
            throw "WallMan exited while opening $SourceName (exit code $($Process.ExitCode))."
        }

        # Some invalid ODB files are checked before their document window is shown.
        # WallMan then asks whether it should open the incorrect database anyway.
        # Continuing is required so that Edit -> Check Database can repair the
        # in-memory copy. Match the full warning text and only click its Yes button.
        $actedOnOpenDialog = $false
        foreach ($dialog in ([WallManNative]::TopWindows($Process.Id) | Where-Object { $_.ClassName -eq '#32770' })) {
            $warningText = (([WallManNative]::ChildWindows($dialog.Handle) | Where-Object {
                $_.ClassName -eq 'Static' -and -not [string]::IsNullOrWhiteSpace($_.Text)
            } | ForEach-Object { $_.Text.Trim() }) -join "`n")
            # WallMan reuses the same native dialog handle for consecutive
            # questions. Include the text so a later, different prompt is not
            # mistaken for an already answered one.
            $dialogKey = '{0}:{1}' -f $dialog.Handle.ToInt64(), $warningText
            if ($answeredDialogs.Contains($dialogKey)) {
                continue
            }

            if ($warningText.IndexOf('error', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $warningText.IndexOf('while loading the database', [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $okButton = [WallManNative]::DialogItem($dialog.Handle, 1)
                if ($okButton -ne [IntPtr]::Zero) {
                    [WallManNative]::Click($okButton)
                }
                throw "WallMan reported an error while loading the database: $SourceName"
            }

            if ($warningText.IndexOf('Not possible to determine intersection of two lines', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $warningText.IndexOf('indentical points', [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $okButton = [WallManNative]::DialogItem($dialog.Handle, 2)
                if ($okButton -eq [IntPtr]::Zero) {
                    throw "WallMan reported a degenerate line, but its OK button was not found: $SourceName"
                }
                [WallManNative]::Click($okButton)
                $actedOnOpenDialog = $true
                continue
            }

            if ($warningText.IndexOf('Database not correct!', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $warningText.IndexOf('Do you wish to continue anyway?', [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $yesButton = [WallManNative]::DialogItem($dialog.Handle, 6)
                if ($yesButton -eq [IntPtr]::Zero) {
                    throw "WallMan reported an incorrect database, but its Yes button was not found: $SourceName"
                }
                [void]$answeredDialogs.Add($dialogKey)
                Write-Verbose "WallMan reported an incorrect database; continuing so it can be repaired: $SourceName"
                [WallManNative]::Click($yesButton)
                $actedOnOpenDialog = $true
                continue
            }

            if ($warningText.IndexOf('lower left corner of this database is negative', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $warningText.IndexOf('move all buildings of the database to positive coordinates', [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $yesButton = [WallManNative]::DialogItem($dialog.Handle, 6)
                if ($yesButton -eq [IntPtr]::Zero) {
                    throw "WallMan requested conversion to positive coordinates, but its Yes button was not found: $SourceName"
                }
                [void]$answeredDialogs.Add($dialogKey)
                Write-Verbose "Moving the checked copy to positive coordinates as required for preprocessing: $SourceName"
                [WallManNative]::Click($yesButton)
                $actedOnOpenDialog = $true
                continue
            }

            if ($dialog.Text.StartsWith('Checking Database', [System.StringComparison]::OrdinalIgnoreCase)) {
                $progressControl = @([WallManNative]::ChildWindows($dialog.Handle) | Where-Object {
                    $_.ClassName -eq 'msctls_progress32' -and $_.ControlId -eq 1173
                } | Select-Object -First 1)
                if ($progressControl.Count -gt 0 -and
                    [WallManNative]::ProgressPosition($progressControl[0].Handle) -ge 100) {
                    $closeButton = [WallManNative]::DialogItem($dialog.Handle, 1214)
                    if ($closeButton -ne [IntPtr]::Zero) {
                        Write-Verbose "WallMan database check reached 100%; closing its result window: $SourceName"
                        [WallManNative]::Click($closeButton)
                        $actedOnOpenDialog = $true
                    }
                }
            }
        }

        if ($actedOnOpenDialog) {
            Start-Sleep -Milliseconds 100
            continue
        }

        $mainWindow = Get-MainWindow -WallManProcessId $Process.Id -SourceName $SourceName
        if ($mainWindow -ne [IntPtr]::Zero) {
            Wait-WallManReady `
                -Process $Process `
                -MainWindow $mainWindow `
                -SourceName $SourceName `
                -TimeoutSec $TimeoutSec
            return $mainWindow
        }

        Start-Sleep -Milliseconds 100
    } while ((Get-Date) -lt $deadline)

    throw "Timed out after $TimeoutSec seconds while opening $SourceName."
}

function Wait-WallManReady {
    param(
        [System.Diagnostics.Process]$Process,
        [IntPtr]$MainWindow,
        [string]$SourceName,
        [int]$TimeoutSec,
        [int]$DiagnosticPauseSec = 0
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $readySince = $null
    $answeredDialogs = New-Object 'System.Collections.Generic.HashSet[string]'
    $diagnosedProgressDialogs = New-Object 'System.Collections.Generic.HashSet[string]'

    do {
        $Process.Refresh()
        if ($Process.HasExited) {
            throw "WallMan exited while checking $SourceName (exit code $($Process.ExitCode))."
        }

        $windows = @([WallManNative]::TopWindows($Process.Id))
        $actedOnDialog = $false
        foreach ($dialog in ($windows | Where-Object { $_.ClassName -eq '#32770' -and $_.Visible })) {
            $warningText = (([WallManNative]::ChildWindows($dialog.Handle) | Where-Object {
                $_.ClassName -eq 'Static' -and -not [string]::IsNullOrWhiteSpace($_.Text)
            } | ForEach-Object { $_.Text.Trim() }) -join "`n")
            $dialogKey = '{0}:{1}' -f $dialog.Handle.ToInt64(), $warningText
            if ($answeredDialogs.Contains($dialogKey)) {
                continue
            }

            if ($warningText.IndexOf('error', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $warningText.IndexOf('while loading the database', [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $okButton = [WallManNative]::DialogItem($dialog.Handle, 1)
                if ($okButton -ne [IntPtr]::Zero) {
                    [WallManNative]::Click($okButton)
                }
                throw "WallMan reported an error while loading the database: $SourceName"
            }

            if ($warningText.IndexOf('Not possible to determine intersection of two lines', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $warningText.IndexOf('indentical points', [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $okButton = [WallManNative]::DialogItem($dialog.Handle, 2)
                if ($okButton -eq [IntPtr]::Zero) {
                    throw "WallMan reported a degenerate line, but its OK button was not found: $SourceName"
                }
                [WallManNative]::Click($okButton)
                $actedOnDialog = $true
                continue
            }

            if ($warningText.IndexOf('Database not correct!', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $warningText.IndexOf('Do you wish to continue anyway?', [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $yesButton = [WallManNative]::DialogItem($dialog.Handle, 6)
                if ($yesButton -eq [IntPtr]::Zero) {
                    throw "WallMan reported an incorrect database, but its Yes button was not found: $SourceName"
                }
                [void]$answeredDialogs.Add($dialogKey)
                Write-Verbose "WallMan reported an incorrect database; continuing so it can be repaired: $SourceName"
                [WallManNative]::Click($yesButton)
                $actedOnDialog = $true
                continue
            }

            if ($warningText.IndexOf('lower left corner of this database is negative', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $warningText.IndexOf('move all buildings of the database to positive coordinates', [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $yesButton = [WallManNative]::DialogItem($dialog.Handle, 6)
                if ($yesButton -eq [IntPtr]::Zero) {
                    throw "WallMan requested conversion to positive coordinates, but its Yes button was not found: $SourceName"
                }
                [void]$answeredDialogs.Add($dialogKey)
                Write-Verbose "Moving the checked copy to positive coordinates as required for preprocessing: $SourceName"
                [WallManNative]::Click($yesButton)
                $actedOnDialog = $true
                continue
            }

            # WallMan leaves its database-check result window open at 100% and
            # requires the Close button before the document becomes usable.
            if ($dialog.Text.StartsWith('Checking Database', [System.StringComparison]::OrdinalIgnoreCase)) {
                $progressControl = @([WallManNative]::ChildWindows($dialog.Handle) | Where-Object {
                    $_.ClassName -eq 'msctls_progress32' -and $_.ControlId -eq 1173
                } | Select-Object -First 1)
                if ($progressControl.Count -gt 0 -and
                    [WallManNative]::ProgressPosition($progressControl[0].Handle) -ge 100) {
                    $progressKey = [string]$dialog.Handle.ToInt64()
                    if ($DiagnosticPauseSec -gt 0 -and
                        $diagnosedProgressDialogs.Add($progressKey)) {
                        Write-Host "WallMan completed its database check for $SourceName."
                        Write-Host "Diagnostic controls in '$($dialog.Text)':"
                        foreach ($child in [WallManNative]::ChildWindows($dialog.Handle)) {
                            Write-Host (
                                '  id={0}; class={1}; text=<{2}>' -f
                                $child.ControlId,
                                $child.ClassName,
                                $child.Text.Replace("`r", '\r').Replace("`n", '\n')
                            )
                            if ($child.ClassName -eq 'ListBox') {
                                foreach ($item in [WallManNative]::ListBoxItems($child.Handle)) {
                                    Write-Host "    item=<$item>"
                                }
                            }
                        }
                        Start-Sleep -Seconds $DiagnosticPauseSec
                    }
                    $closeButton = [WallManNative]::DialogItem($dialog.Handle, 1214)
                    if ($closeButton -ne [IntPtr]::Zero) {
                        Write-Verbose "WallMan database check reached 100%; closing its result window: $SourceName"
                        [WallManNative]::Click($closeButton)
                        $actedOnDialog = $true
                    }
                }
            }
        }

        if ($actedOnDialog) {
            $readySince = $null
            Start-Sleep -Milliseconds 100
            continue
        }

        $blockingDialogs = @($windows | Where-Object { $_.ClassName -eq '#32770' -and $_.Visible })
        $mainInfo = @($windows | Where-Object { $_.Handle -eq $MainWindow } | Select-Object -First 1)
        if ($mainInfo.Count -gt 0 -and $mainInfo[0].Enabled -and $blockingDialogs.Count -eq 0) {
            if ($null -eq $readySince) {
                $readySince = Get-Date
            }
            elseif (((Get-Date) - $readySince).TotalMilliseconds -ge 250) {
                return
            }
        }
        else {
            $readySince = $null
        }

        Start-Sleep -Milliseconds 50
    } while ((Get-Date) -lt $deadline)

    $details = Get-DialogSummary -WallManProcessId $Process.Id
    throw "WallMan did not become ready within $TimeoutSec seconds while checking $SourceName. $details"
}

function Get-DialogSummary {
    param([int]$WallManProcessId)

    $summaries = New-Object System.Collections.Generic.List[string]
    foreach ($dialog in ([WallManNative]::TopWindows($WallManProcessId) | Where-Object { $_.ClassName -eq '#32770' })) {
        $parts = New-Object System.Collections.Generic.List[string]
        foreach ($child in [WallManNative]::ChildWindows($dialog.Handle)) {
            if ($child.ClassName -eq 'Static' -and -not [string]::IsNullOrWhiteSpace($child.Text)) {
                $parts.Add($child.Text.Trim())
            }
        }
        $text = ($parts -join ' | ')
        $summaries.Add("$($dialog.Text): $text")
    }

    if ($summaries.Count -eq 0) {
        return 'No dialog was found.'
    }
    return ($summaries -join ' || ')
}

function Wait-SaveDialog {
    param(
        [int]$WallManProcessId,
        [int]$TimeoutSec
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    do {
        foreach ($dialog in ([WallManNative]::TopWindows($WallManProcessId) | Where-Object { $_.ClassName -eq '#32770' })) {
            $fileNameEdit = @([WallManNative]::ChildWindows($dialog.Handle) | Where-Object {
                $_.ClassName -eq 'Edit' -and $_.ControlId -eq 1148
            } | Select-Object -First 1)
            $saveButton = [WallManNative]::DialogItem($dialog.Handle, 1)

            if ($fileNameEdit.Count -gt 0 -and $saveButton -ne [IntPtr]::Zero) {
                return [pscustomobject]@{
                    Dialog = $dialog.Handle
                    FileNameEdit = $fileNameEdit[0].Handle
                    SaveButton = $saveButton
                }
            }
        }

        Start-Sleep -Milliseconds 50
    } while ((Get-Date) -lt $deadline)

    $details = Get-DialogSummary -WallManProcessId $WallManProcessId
    throw "Save As dialog did not appear within $TimeoutSec seconds. $details"
}

function Wait-SaveFinished {
    param(
        [System.Diagnostics.Process]$Process,
        [IntPtr]$SaveDialog,
        [string]$TemporaryOutput,
        [string]$SaveFileName,
        [string]$SourceName,
        [int]$TimeoutSec
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $savedWithDatabaseWarning = $false
    $answeredDialogs = New-Object 'System.Collections.Generic.HashSet[string]'
    $resubmitCount = 0
    $nextResubmit = (Get-Date).AddMilliseconds(800)
    do {
        $Process.Refresh()
        if ($Process.HasExited) {
            throw "WallMan exited while saving $SourceName (exit code $($Process.ExitCode))."
        }

        $actedOnDialog = $false
        foreach ($dialog in ([WallManNative]::TopWindows($Process.Id) | Where-Object { $_.ClassName -eq '#32770' -and $_.Visible })) {
            $dialogText = (([WallManNative]::ChildWindows($dialog.Handle) | Where-Object {
                $_.ClassName -eq 'Static' -and -not [string]::IsNullOrWhiteSpace($_.Text)
            } | ForEach-Object { $_.Text.Trim() }) -join "`n")
            $dialogKey = '{0}:{1}' -f $dialog.Handle.ToInt64(), $dialogText

            if ($dialogText.IndexOf('Not possible to determine intersection of two lines', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $dialogText.IndexOf('indentical points', [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $okButton = [WallManNative]::DialogItem($dialog.Handle, 2)
                if ($okButton -eq [IntPtr]::Zero) {
                    throw "WallMan reported a degenerate line while saving, but its OK button was not found: $SourceName"
                }
                [WallManNative]::Click($okButton)
                $actedOnDialog = $true
                continue
            }

            if (-not $answeredDialogs.Contains($dialogKey) -and
                $dialogText.IndexOf('lower left corner of this database is negative', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $dialogText.IndexOf('move all buildings of the database to positive coordinates', [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $yesButton = [WallManNative]::DialogItem($dialog.Handle, 6)
                if ($yesButton -eq [IntPtr]::Zero) {
                    throw "WallMan requested conversion to positive coordinates while saving, but its Yes button was not found: $SourceName"
                }
                [void]$answeredDialogs.Add($dialogKey)
                Write-Verbose "Moving the checked copy to positive coordinates while saving: $SourceName"
                [WallManNative]::Click($yesButton)
                $actedOnDialog = $true
                continue
            }

            # A database that WallMan could not fully repair can still be saved as
            # a separate checked file. Record this explicitly for downstream filtering.
            if (-not $answeredDialogs.Contains($dialogKey) -and
                $dialogText.IndexOf('The database might contain errors!', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $dialogText.IndexOf('Do you want to save it anyway?', [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $yesButton = [WallManNative]::DialogItem($dialog.Handle, 6)
                if ($yesButton -eq [IntPtr]::Zero) {
                    throw "WallMan displayed its save-with-errors warning, but its Yes button was not found: $SourceName"
                }
                [void]$answeredDialogs.Add($dialogKey)
                $savedWithDatabaseWarning = $true
                Write-Verbose "WallMan could not fully repair the database; saving a flagged checked copy: $SourceName"
                [WallManNative]::Click($yesButton)
                $actedOnDialog = $true
                continue
            }

            if ($dialog.Text.StartsWith('Checking Database', [System.StringComparison]::OrdinalIgnoreCase)) {
                $progressControl = @([WallManNative]::ChildWindows($dialog.Handle) | Where-Object {
                    $_.ClassName -eq 'msctls_progress32' -and $_.ControlId -eq 1173
                } | Select-Object -First 1)
                if ($progressControl.Count -gt 0 -and
                    [WallManNative]::ProgressPosition($progressControl[0].Handle) -ge 100) {
                    $closeButton = [WallManNative]::DialogItem($dialog.Handle, 1214)
                    if ($closeButton -ne [IntPtr]::Zero) {
                        [WallManNative]::Click($closeButton)
                        $actedOnDialog = $true
                    }
                }
            }
        }

        if ($actedOnDialog) {
            Start-Sleep -Milliseconds 100
            continue
        }

        if ((Test-Path -LiteralPath $TemporaryOutput -PathType Leaf) -and
            -not [WallManNative]::Exists($SaveDialog)) {
            return $savedWithDatabaseWarning
        }

        # The legacy Windows common dialog can asynchronously clear a filename
        # that was written during shell initialization. If the original Save As
        # dialog is still open, refill the exact transactional path and submit it
        # again instead of occupying the shared save lock until timeout.
        if ([WallManNative]::Exists($SaveDialog) -and
            (Get-Date) -ge $nextResubmit -and
            $resubmitCount -lt 5) {
            $fileNameEdit = @([WallManNative]::ChildWindows($SaveDialog) | Where-Object {
                $_.ClassName -eq 'Edit' -and $_.ControlId -eq 1148
            } | Select-Object -First 1)
            $saveButton = [WallManNative]::DialogItem($SaveDialog, 1)

            if ($fileNameEdit.Count -gt 0 -and $saveButton -ne [IntPtr]::Zero) {
                if ([WallManNative]::SetText($fileNameEdit[0].Handle, $SaveFileName)) {
                    Start-Sleep -Milliseconds 100
                    [WallManNative]::Click($saveButton)
                    $resubmitCount++
                    Write-Verbose "Re-submitted a Save As filename cleared by the common dialog ($resubmitCount/5): $SourceName"
                }
            }
            $nextResubmit = (Get-Date).AddMilliseconds(800)
            continue
        }
        Start-Sleep -Milliseconds 100
    } while ((Get-Date) -lt $deadline)

    throw "WallMan did not finish saving within $TimeoutSec seconds: $TemporaryOutput"
}

function Close-WallMan {
    param(
        [System.Diagnostics.Process]$Process,
        [IntPtr]$MainWindow
    )

    if ($null -eq $Process) {
        return
    }

    $Process.Refresh()
    if ($Process.HasExited) {
        return
    }

    if ($MainWindow -ne [IntPtr]::Zero -and [WallManNative]::Exists($MainWindow)) {
        [void][WallManNative]::PostClose($MainWindow)
    }

    if (-not $Process.WaitForExit(5000)) {
        # This process was started by this script. A forced stop here only discards
        # its unsaved in-memory copy; the source ODB is never modified.
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        [void]$Process.WaitForExit(5000)
    }
}

function Wait-FileReady {
    param(
        [string]$Path,
        [int]$TimeoutSec
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    do {
        if (Test-Path -LiteralPath $Path -PathType Leaf) {
            try {
                $stream = [System.IO.File]::Open(
                    $Path,
                    [System.IO.FileMode]::Open,
                    [System.IO.FileAccess]::Read,
                    [System.IO.FileShare]::None)
                try {
                    if ($stream.Length -gt 0) {
                        return $stream.Length
                    }
                }
                finally {
                    $stream.Dispose()
                }
            }
            catch [System.IO.IOException] {
                # WallMan may still be flushing or closing the file.
            }
        }
        Start-Sleep -Milliseconds 100
    } while ((Get-Date) -lt $deadline)

    throw "Saved file is missing, empty, or still locked: $Path"
}

function Invoke-CheckOneDatabase {
    param(
        [string]$SourcePath,
        [string]$OutputPath
    )

    $sourceItem = Get-Item -LiteralPath $SourcePath
    # WallMan 2020's legacy Save As dialog can fail on long full paths. Keep the
    # transactional filename deliberately short; it is renamed after WallMan exits.
    $shortId = [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $temporaryName = '__wmchk_{0}.odb' -f $shortId
    $temporaryOutput = Join-Path $sourceItem.DirectoryName $temporaryName
    $wallManProcess = $null
    $mainWindow = [IntPtr]::Zero
    $saveSemaphore = $null
    $saveSemaphoreHeld = $false
    $saveCompleted = $false
    $savedWithDatabaseWarning = $false

    try {
        try {
            $quotedSource = '"' + $sourceItem.FullName + '"'
            $wallManProcess = Start-Process `
                -FilePath $resolvedWallMan `
                -ArgumentList $quotedSource `
                -WindowStyle Hidden `
                -PassThru

            $mainWindow = Wait-MainWindow `
                -Process $wallManProcess `
                -SourceName $sourceItem.Name `
                -TimeoutSec $OpenTimeoutSec

            # WallMan menu: Edit -> Check Database (Ctrl+K), command ID 32958.
            # Post asynchronously so the watcher below remains able to answer
            # WallMan's modal validation prompt while the command is running.
            if (-not [WallManNative]::PostCommand($mainWindow, 32958)) {
                throw 'Could not invoke Check Database.'
            }

            # The MFC command can return before its progress dialog and validation
            # prompt have completed. Wait until WallMan is truly idle before saving.
            Wait-WallManReady `
                -Process $wallManProcess `
                -MainWindow $mainWindow `
                -SourceName $sourceItem.Name `
                -TimeoutSec $CheckTimeoutSec `
                -DiagnosticPauseSec $DiagnosticPauseSec

            # WallMan 2020's legacy Save As dialog becomes unreliable when every
            # worker opens it at once. Allow a small, bounded amount of save
            # parallelism instead of serializing all workers behind one mutex.
            $saveSemaphore = New-Object System.Threading.Semaphore(
                $SaveConcurrency,
                $SaveConcurrency,
                'Local\CodexWallManSaveDatabaseV2'
            )
            $semaphoreWaitSec = [math]::Max(120, $SaveTimeoutSec * $ShardCount)
            $saveSemaphoreHeld = $saveSemaphore.WaitOne(
                [TimeSpan]::FromSeconds($semaphoreWaitSec)
            )
            if (-not $saveSemaphoreHeld) {
                throw "Timed out while waiting for a WallMan save slot after $semaphoreWaitSec seconds."
            }

            # WallMan menu: File -> Save Database As, command ID 57604.
            if (-not [WallManNative]::PostCommand($mainWindow, 57604)) {
                throw 'Could not invoke Save Database As.'
            }

            $saveDialog = Wait-SaveDialog `
                -WallManProcessId $wallManProcess.Id `
                -TimeoutSec $SaveTimeoutSec

            # The common dialog creates its controls before shell initialization is
            # completely idle. Writing the filename during that small window can be
            # discarded by the dialog, so wait briefly and then submit it.
            [void]$wallManProcess.WaitForInputIdle(1000)
            Start-Sleep -Milliseconds 150

            # Always use the absolute transactional path. WallMan's legacy
            # dialog remembers the last folder globally across its processes,
            # so a filename alone can silently save into a different tile.
            if (-not [WallManNative]::SetText($saveDialog.FileNameEdit, $temporaryOutput)) {
                throw 'Could not set the Save As filename.'
            }
            Start-Sleep -Milliseconds 100
            # Write once more after the common dialog's asynchronous shell update.
            # This is cheap and prevents most blank-filename save stalls.
            if ([WallManNative]::Exists($saveDialog.Dialog)) {
                [void][WallManNative]::SetText($saveDialog.FileNameEdit, $temporaryOutput)
            }
            Start-Sleep -Milliseconds 50
            [WallManNative]::Click($saveDialog.SaveButton)

            $savedWithDatabaseWarning = Wait-SaveFinished `
                -Process $wallManProcess `
                -SaveDialog $saveDialog.Dialog `
                -TemporaryOutput $temporaryOutput `
                -SaveFileName $temporaryOutput `
                -SourceName $sourceItem.Name `
                -TimeoutSec $SaveTimeoutSec
            $saveCompleted = $true
        }
        finally {
            # Once the output exists and the Save As dialog is gone, let the next
            # worker enter its save section before this WallMan process closes.
            # On errors, close the blocking dialog/process first and only then
            # release the slot.
            if ($saveCompleted -and $saveSemaphoreHeld -and $null -ne $saveSemaphore) {
                [void]$saveSemaphore.Release()
                $saveSemaphoreHeld = $false
            }
            if ($saveCompleted -and $null -ne $saveSemaphore) {
                $saveSemaphore.Dispose()
                $saveSemaphore = $null
            }

            Close-WallMan -Process $wallManProcess -MainWindow $mainWindow

            if ($saveSemaphoreHeld -and $null -ne $saveSemaphore) {
                try {
                    [void]$saveSemaphore.Release()
                }
                catch [System.Threading.SemaphoreFullException] {}
            }
            if ($null -ne $saveSemaphore) {
                $saveSemaphore.Dispose()
            }
        }

        $outputSize = Wait-FileReady -Path $temporaryOutput -TimeoutSec $SaveTimeoutSec

        # Never replace a result produced by this or another concurrently running job.
        if (Test-Path -LiteralPath $OutputPath) {
            Remove-Item -LiteralPath $temporaryOutput -Force -ErrorAction SilentlyContinue
            return [pscustomobject]@{
                Status = 'SKIPPED_RACE'
                Size = 0
                Message = 'Output appeared while this file was being processed; existing output was kept.'
            }
        }

        Move-Item -LiteralPath $temporaryOutput -Destination $OutputPath
        return [pscustomobject]@{
            Status = if ($savedWithDatabaseWarning) { 'OK_DATABASE_WARNING' } else { 'OK' }
            Size = $outputSize
            Message = if ($savedWithDatabaseWarning) {
                'WallMan saved the checked copy but reported that the database might still contain errors and cannot be used for prediction.'
            }
            else {
                ''
            }
        }
    }
    catch {
        if (Test-Path -LiteralPath $temporaryOutput) {
            Remove-Item -LiteralPath $temporaryOutput -Force -ErrorAction SilentlyContinue
        }
        throw
    }
}

function Invoke-CheckDatabaseWithRetry {
    param(
        [string]$SourcePath,
        [string]$OutputPath
    )

    $lastException = $null
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        try {
            $result = Invoke-CheckOneDatabase -SourcePath $SourcePath -OutputPath $OutputPath
            if ($attempt -gt 1) {
                $retryNote = "Succeeded on attempt $attempt of $MaxAttempts."
                if ([string]::IsNullOrWhiteSpace($result.Message)) {
                    $result.Message = $retryNote
                }
                else {
                    $result.Message = "$retryNote $($result.Message)"
                }
            }
            return $result
        }
        catch {
            $lastException = $_
            if ($attempt -ge $MaxAttempts) {
                throw
            }
            Write-Warning "Attempt $attempt/$MaxAttempts failed for $SourcePath -- $($_.Exception.Message); retrying."
            if ($RetryDelaySec -gt 0) {
                Start-Sleep -Seconds $RetryDelaySec
            }
        }
    }

    throw $lastException
}

function ConvertTo-CsvField {
    param([object]$Value)
    $text = if ($null -eq $Value) { '' } else { [string]$Value }
    return '"' + $text.Replace('"', '""') + '"'
}

function Write-LogRow {
    param(
        [System.IO.StreamWriter]$Writer,
        [datetime]$Timestamp,
        [string]$Source,
        [string]$Output,
        [string]$Status,
        [string]$Message,
        [double]$DurationSeconds,
        [long]$OutputBytes
    )

    $values = @(
        $Timestamp.ToString('yyyy-MM-dd HH:mm:ss'),
        $Source,
        $Output,
        $Status,
        $Message,
        $DurationSeconds.ToString('0.000', [System.Globalization.CultureInfo]::InvariantCulture),
        $OutputBytes
    )
    $Writer.WriteLine((($values | ForEach-Object { ConvertTo-CsvField $_ }) -join ','))
    $Writer.Flush()
}

function Get-OutputPathForSource {
    param([System.IO.FileInfo]$Source)
    $outputDirectory = if ([string]::IsNullOrWhiteSpace($resolvedOutputRoot)) {
        $Source.DirectoryName
    }
    else {
        $resolvedOutputRoot
    }
    return Join-Path $outputDirectory ($Source.BaseName + $OutputSuffix + '.odb')
}

$sourceFiles = if ([string]::IsNullOrWhiteSpace($SourceListPath)) {
    @(
        Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -Filter '*.odb' |
            Where-Object {
                -not $_.BaseName.EndsWith('_checked', [System.StringComparison]::OrdinalIgnoreCase) -and
                $_.Name -notlike '*.__wallman_tmp_*.odb' -and
                $_.Name -notlike '__wmchk_????????.odb'
            } |
            Sort-Object FullName
    )
}
else {
    @(
        Get-Content -LiteralPath $SourceListPath -Encoding UTF8 |
            ForEach-Object { $_.Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and -not $_.StartsWith('#') } |
            ForEach-Object {
                if (-not (Test-Path -LiteralPath $_ -PathType Leaf)) {
                    throw "A source listed in SourceListPath does not exist: $_"
                }
                Get-Item -LiteralPath $_
            } |
            Sort-Object FullName -Unique
    )
}
$sourceFiles = @($sourceFiles)

if ($ShardCount -gt 1) {
    $shardedFiles = New-Object System.Collections.Generic.List[System.IO.FileInfo]
    for ($sourceIndex = 0; $sourceIndex -lt $sourceFiles.Count; $sourceIndex++) {
        if (($sourceIndex % $ShardCount) -eq $ShardIndex) {
            $shardedFiles.Add($sourceFiles[$sourceIndex])
        }
    }
    $sourceFiles = @($shardedFiles)
}

if ($Limit -gt 0) {
    # A limited smoke test should process pending files, rather than spend its
    # allowance on outputs that already exist from an earlier run.
    $sourceFiles = @($sourceFiles | Where-Object {
        $candidateOutput = Get-OutputPathForSource -Source $_
        -not (Test-Path -LiteralPath $candidateOutput -PathType Leaf)
    } | Select-Object -First $Limit)
}

if ($sourceFiles.Count -eq 0) {
    Write-Host "No matching pending source ODB files were found under: $resolvedRoot"
    exit 0
}

Write-Host "Source root : $resolvedRoot"
Write-Host "WallMan     : $resolvedWallMan"
Write-Host "Source ODBs : $($sourceFiles.Count)"
Write-Host "Shard       : $($ShardIndex + 1) / $ShardCount"
Write-Host "Output rule  : <source name>$OutputSuffix.odb (existing output is skipped)"
if (-not [string]::IsNullOrWhiteSpace($resolvedOutputRoot)) {
    Write-Host "Output root  : $resolvedOutputRoot"
}

if ($DryRun) {
    foreach ($source in $sourceFiles) {
        $output = Get-OutputPathForSource -Source $source
        $state = if (Test-Path -LiteralPath $output) { 'SKIP' } else { 'CHECK' }
        Write-Output ("[{0}] {1} -> {2}" -f $state, $source.FullName, $output)
    }
    exit 0
}

if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $defaultLogRoot = if ([string]::IsNullOrWhiteSpace($resolvedOutputRoot)) {
        $resolvedRoot
    }
    else {
        $resolvedOutputRoot
    }
    if ($ShardCount -gt 1) {
        $workerLogName = '_wallman_check_database_worker_{0:D2}_of_{1:D2}.csv' -f ($ShardIndex + 1), $ShardCount
        $LogPath = Join-Path $defaultLogRoot $workerLogName
    }
    else {
        $LogPath = Join-Path $defaultLogRoot '_wallman_check_database_log.csv'
    }
}
$logParent = Split-Path -Parent $LogPath
if (-not [string]::IsNullOrWhiteSpace($logParent)) {
    New-Item -ItemType Directory -Path $logParent -Force | Out-Null
}
$logExists = Test-Path -LiteralPath $LogPath
$utf8 = New-Object System.Text.UTF8Encoding($true)
$logWriter = New-Object System.IO.StreamWriter($LogPath, $true, $utf8)
if (-not $logExists -or (Get-Item -LiteralPath $LogPath).Length -eq 0) {
    $logWriter.WriteLine('"Timestamp","Source","Output","Status","Message","DurationSeconds","OutputBytes"')
    $logWriter.Flush()
}

$okCount = 0
$warningCount = 0
$skipCount = 0
$failCount = 0
$batchStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

try {
    for ($index = 0; $index -lt $sourceFiles.Count; $index++) {
        $source = $sourceFiles[$index]
        $output = Get-OutputPathForSource -Source $source
        $position = $index + 1
        $percent = [int](100.0 * $position / $sourceFiles.Count)
        Write-Progress `
            -Activity 'WallMan Check Database' `
            -Status "$position / $($sourceFiles.Count): $($source.Name)" `
            -PercentComplete $percent

        if (Test-Path -LiteralPath $output -PathType Leaf) {
            $skipCount++
            $existingBytes = (Get-Item -LiteralPath $output).Length
            Write-LogRow `
                -Writer $logWriter `
                -Timestamp (Get-Date) `
                -Source $source.FullName `
                -Output $output `
                -Status 'SKIPPED_EXISTS' `
                -Message 'Existing output was kept.' `
                -DurationSeconds 0 `
                -OutputBytes $existingBytes
            continue
        }

        $itemStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $result = Invoke-CheckDatabaseWithRetry -SourcePath $source.FullName -OutputPath $output
            $itemStopwatch.Stop()

            if ($result.Status.StartsWith('OK', [System.StringComparison]::OrdinalIgnoreCase)) {
                $okCount++
                if ($result.Status -eq 'OK_DATABASE_WARNING') {
                    $warningCount++
                }
            }
            else {
                $skipCount++
            }

            Write-LogRow `
                -Writer $logWriter `
                -Timestamp (Get-Date) `
                -Source $source.FullName `
                -Output $output `
                -Status $result.Status `
                -Message $result.Message `
                -DurationSeconds $itemStopwatch.Elapsed.TotalSeconds `
                -OutputBytes $result.Size
        }
        catch {
            $itemStopwatch.Stop()
            $failCount++
            $message = $_.Exception.Message
            Write-Warning "[$position/$($sourceFiles.Count)] FAILED: $($source.FullName) -- $message"
            Write-LogRow `
                -Writer $logWriter `
                -Timestamp (Get-Date) `
                -Source $source.FullName `
                -Output $output `
                -Status 'FAILED' `
                -Message $message `
                -DurationSeconds $itemStopwatch.Elapsed.TotalSeconds `
                -OutputBytes 0

            if ($StopOnError) {
                throw
            }
        }
    }
}
finally {
    Write-Progress -Activity 'WallMan Check Database' -Completed
    $batchStopwatch.Stop()
    $logWriter.Dispose()
}

Write-Host ''
Write-Host 'Batch finished.'
Write-Host "Created : $okCount"
Write-Host "Warnings: $warningCount"
Write-Host "Skipped : $skipCount"
Write-Host "Failed  : $failCount"
Write-Host ("Elapsed : {0}" -f $batchStopwatch.Elapsed)
Write-Host "Log     : $LogPath"

if ($failCount -gt 0) {
    exit 2
}
