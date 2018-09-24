const electron = require('electron');

const {app, BrowserWindow, Menu, ipcMain} = electron;

let mainWindow;
let addWindow;

app.on('ready', () => {
    mainWindow = new BrowserWindow({});
    mainWindow.loadURL(`file://${__dirname}/main.html`);
    mainWindow.on('closed', () => app.quit());

    const mainMenu = Menu.buildFromTemplate(menuTemplate);
    Menu.setApplicationMenu(mainMenu);
});

ipcMain.on('todo:add', (event, todo) => {
    mainWindow.webContents.send('todo:add', todo);
    addWindow.close();
});

function createAddWindow() {
    addWindow = new BrowserWindow({
        width: 300,
        height: 200,
        title: 'Add New todo'
    });
    addWindow.loadURL(`file://${__dirname}/add.html`);
    addWindow.on('closed', () => addWindow = null);
}

const menuTemplate = [
    {
        label: 'File',
        submenu: [
            {
                label: 'New Todo',
                click() {
                    createAddWindow();
                }
            },
            {
                label: 'Clear Todos',
                click() {
                    // mainWindow.reload();
                    mainWindow.webContents.send('todo:clear');
                }
            },
            {
                label: 'Quit',
                accelerator: process.platform === 'darwin' ? 'Command+Q' : 'Control+Q',
                click() {
                    app.quit();
                }
            }
        ]
    }
];

if (process.platform === 'darwin') {
    menuTemplate.unshift({});
}

// production, development, staging, test
if (process.env.NODE_ENV !== 'production') {
    menuTemplate.push({
        label: 'Developer',
        submenu: [
            {
                role: 'reload'
            },
            {
                label: 'Toggle Developer Tools',
                accelerator: process.platform === 'darwin' ? 'Command+Alt+I' : 'Control+Shift+I',
                click(item, focusedWindow) {
                    focusedWindow.toggleDevTools();
                }
            }
        ]
    })
}