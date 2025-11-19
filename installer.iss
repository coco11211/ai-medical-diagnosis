; Trading Bot Simulator - Inno Setup Installer Script
; This script creates a professional Windows installer

#define MyAppName "Trading Bot Simulator"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Trading Bot Simulator"
#define MyAppURL "https://github.com/yourusername/trading-bot-simulator"
#define MyAppExeName "TradingBotSimulator.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application
AppId={{A8F3D9E2-1C4B-4A5D-9E6F-3B7C8D9E1F2A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=LICENSE.txt
; Uncomment the following line if you have an info file before installation
; InfoBeforeFile=README.txt
OutputDir=Output
OutputBaseFilename=TradingBotSimulator-Setup
SetupIconFile=resources\icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
Source: "dist\TradingBotSimulator\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\TradingBotSimulator\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "config.yaml"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:ProgramOnTheWeb,{#MyAppName}}"; Filename: "{#MyAppURL}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
var
  ErrorCode: Integer;
  PythonInstalled: Boolean;
begin
  Result := True;

  // Check if application is already running
  if CheckForMutexes('TradingBotSimulatorMutex') then
  begin
    if MsgBox('Trading Bot Simulator is currently running. Please close it before continuing installation.' + #13#10#13#10 + 'Continue anyway?', mbConfirmation, MB_YESNO) = IDNO then
    begin
      Result := False;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Perform any post-installation tasks here
  end;
end;

function InitializeUninstall(): Boolean;
begin
  Result := True;

  // Check if application is running before uninstall
  if CheckForMutexes('TradingBotSimulatorMutex') then
  begin
    MsgBox('Trading Bot Simulator is currently running. Please close it before uninstalling.', mbError, MB_OK);
    Result := False;
  end;
end;
