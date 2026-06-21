# run_cluster.ps1  -  Windows PowerShell 클러스터 관리자
# ============================================================
# N개의 vlm_moondream.py 서버를 포트 8000~(8000+N-1)에 띄움.
#
# 사용법:
#   powershell -ExecutionPolicy Bypass -File .\run_cluster.ps1 start 1
#   powershell -ExecutionPolicy Bypass -File .\run_cluster.ps1 start 2
#   powershell -ExecutionPolicy Bypass -File .\run_cluster.ps1 start 3
#   powershell -ExecutionPolicy Bypass -File .\run_cluster.ps1 stop
#   powershell -ExecutionPolicy Bypass -File .\run_cluster.ps1 status
#   powershell -ExecutionPolicy Bypass -File .\run_cluster.ps1 logs 8000
#
# 또는 PowerShell 안에서 실행 정책 한 번 설정하면 더 간편:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# 그 후:
#   .\run_cluster.ps1 start 3

param(
    [Parameter(Position=0)][string]$Action = "status",
    [Parameter(Position=1)][int]$N = 1
)

$LogDir   = "$env:USERPROFILE\.moondream_logs"
$PidDir   = "$env:USERPROFILE\.moondream_pids"
$BasePort = 8000

# 현재 스크립트가 있는 폴더의 vlm_moondream.py 사용
$ScriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$ServerScript = Join-Path $ScriptDir "vlm_moondream.py"
$PythonExe = Join-Path $ScriptDir "venv\Scripts\python.exe"

# 로그/PID 폴더 준비
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $PidDir | Out-Null


function Start-Cluster {
    param([int]$Count)

    if (-not (Test-Path $ServerScript)) {
        Write-Host "❌ vlm_moondream.py 없음: $ServerScript" -ForegroundColor Red
        exit 1
    }

    Write-Host "Launching $Count vlm_moondream servers (base port $BasePort)" -ForegroundColor Cyan
    for ($i = 0; $i -lt $Count; $i++) {
        $port    = $BasePort + $i
        $pidFile = Join-Path $PidDir "moondream_$port.pid"
        $logFile = Join-Path $LogDir "moondream_$port.log"

        # 이미 떠 있는지 확인
        if (Test-Path $pidFile) {
            $oldPid = Get-Content $pidFile
            $proc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Host "  :$port already running (pid $oldPid)" -ForegroundColor Yellow
                continue
            }
        }

        # python vlm_moondream.py --port $port  를 백그라운드로 띄움
        $proc = Start-Process -FilePath $PythonExe `
            -ArgumentList "`"$ServerScript`"", "--port", $port `
            -RedirectStandardOutput $logFile `
            -RedirectStandardError "$logFile.err" `
            -WindowStyle Hidden `
            -PassThru

        $proc.Id | Out-File -FilePath $pidFile -Encoding ascii
        Write-Host "  :$port started (pid $($proc.Id))  log: $logFile" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "모델 로딩 대기 중 (보통 30~120초)." -ForegroundColor Cyan
    Write-Host "상태 확인:  .\run_cluster.ps1 status"
    Write-Host "로그 확인:  .\run_cluster.ps1 logs $BasePort"
}


function Stop-Cluster {
    Write-Host "Stopping all moondream servers..." -ForegroundColor Cyan
    $killed = 0
    Get-ChildItem -Path $PidDir -Filter "moondream_*.pid" -ErrorAction SilentlyContinue | ForEach-Object {
        $pidValue = Get-Content $_.FullName -ErrorAction SilentlyContinue
        if ($pidValue) {
            $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
            if ($proc) {
                Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
                Write-Host "  killed pid $pidValue" -ForegroundColor Yellow
                $killed++
            }
        }
        Remove-Item $_.FullName -ErrorAction SilentlyContinue
    }
    Write-Host "  $killed servers stopped" -ForegroundColor Green

    # 좀비 정리 (vlm_moondream.py 이름이 들어간 python 프로세스 강제 종료)
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -like "*vlm_moondream.py*"
    } | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "  cleaned up zombie pid $($_.ProcessId)" -ForegroundColor DarkYellow
    }
}


function Show-Status {
    Write-Host "moondream servers:" -ForegroundColor Cyan
    $any = $false
    Get-ChildItem -Path $PidDir -Filter "moondream_*.pid" -ErrorAction SilentlyContinue | ForEach-Object {
        $any = $true
        $port = $_.BaseName -replace "moondream_", ""
        $pidValue = Get-Content $_.FullName -ErrorAction SilentlyContinue
        $proc = if ($pidValue) { Get-Process -Id $pidValue -ErrorAction SilentlyContinue } else { $null }

        if ($proc) {
            # 헬스체크
            try {
                $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 1 -ErrorAction Stop
                if ($resp.StatusCode -eq 200) {
                    Write-Host "  :$port  ready    (pid $pidValue)" -ForegroundColor Green
                } else {
                    Write-Host "  :$port  loading  (pid $pidValue)" -ForegroundColor Yellow
                }
            } catch {
                Write-Host "  :$port  loading  (pid $pidValue)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  :$port  dead     (stale pid)" -ForegroundColor Red
            Remove-Item $_.FullName -ErrorAction SilentlyContinue
        }
    }
    if (-not $any) {
        Write-Host "  (none running)" -ForegroundColor DarkGray
    }
}


function Show-Logs {
    param([int]$Port)
    $logFile = Join-Path $LogDir "moondream_$Port.log"
    if (Test-Path $logFile) {
        Write-Host "Tailing $logFile (Ctrl+C to stop)" -ForegroundColor Cyan
        Get-Content $logFile -Wait -Tail 30
    } else {
        Write-Host "❌ 로그 파일 없음: $logFile" -ForegroundColor Red
        Write-Host "사용 가능한 로그:"
        Get-ChildItem $LogDir -ErrorAction SilentlyContinue | Select-Object Name
    }
}


# ── 메인 디스패치 ──
switch ($Action.ToLower()) {
    "start"  { Start-Cluster -Count $N }
    "stop"   { Stop-Cluster }
    "status" { Show-Status }
    "logs"   { Show-Logs -Port $N }
    default {
        Write-Host "Usage: .\run_cluster.ps1 {start <N>|stop|status|logs <PORT>}"
        Write-Host "Examples:"
        Write-Host "  .\run_cluster.ps1 start 1     # 단일 인스턴스 (baseline)"
        Write-Host "  .\run_cluster.ps1 start 3     # 3개 서버 (시연 최종)"
        Write-Host "  .\run_cluster.ps1 stop"
        Write-Host "  .\run_cluster.ps1 status"
        Write-Host "  .\run_cluster.ps1 logs 8000"
        exit 1
    }
}
