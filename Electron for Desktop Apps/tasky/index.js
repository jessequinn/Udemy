const electron = require('electron');
const path = require('path');
const {app, ipcMain} = electron;
const TimerTray = require('./app/timer_tray');
const TimerBrowserWindow = require('./app/timer_browserwindow');

// needed to avoid garbage collector
let mainWindow;
let tray;

app.on('ready', () => {
    app.dock.hide();

    mainWindow = new TimerBrowserWindow({
        height: 500,
        width: 300,
        frame: false,
        resizable: false,
        show: false,
        webPreferences: {
            backgroundThrottling: false
        }
    }, `file://${__dirname}/src/index.html`);

    const iconName = process.platform === 'win32' ? 'windows-icon.png' : 'iconTemplate.png';
    const iconPath = path.join(__dirname, `./src/assets/${iconName}`);

    tray = new TimerTray(iconPath, mainWindow);
});

ipcMain.on('update:timer', (event, timeLeft) => {
    tray.setTitle(timeLeft);
});