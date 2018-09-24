const electron = require('electron');
const ffmpeg = require('fluent-ffmpeg');
const {app, BrowserWindow, ipcMain} = electron;

let mainWindow;

app.on('ready', () => {
    // console.log('App is now ready');
    mainWindow = new BrowserWindow({});
    mainWindow.loadURL(`file://${__dirname}/index.html`);
});

ipcMain.on('video:submit', (event, path) => {
    ffmpeg.ffprobe(path, (err, metadata) => {
        if (err) {
            console.log(`Error happened: ${err}`);
        } else {
            // console.log(`Video duration metadata = ${metadata.format.duration}`);
            mainWindow.webContents.send('video:duration', metadata.format.duration);
        }
    })
});