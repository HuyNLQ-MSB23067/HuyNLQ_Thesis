const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer');
const path = require('path');
const db = require('./db');
const indexRoutes = require('./routes/index');
const detailsRoutes = require('./routes/details');
const WebSocket = require('ws');

const app = express();
const upload = multer({ dest: 'uploads/datasets/' });

app.set('view engine', 'ejs');

app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));  // Serve static files from the 'public' folder
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use('/', indexRoutes);
app.use('/details', detailsRoutes);

const server = app.listen(1000, () => {
    console.log('Server is running on port 1000');
});
const wss = new WebSocket.Server({ server });

const clients = {};

wss.on('connection', (ws, req) => {
    const urlParams = new URLSearchParams(req.url.replace('/?', ''));
    const datasetId = urlParams.get('datasetId');
    
    if (!clients[datasetId]) {
        clients[datasetId] = [];
    }
    clients[datasetId].push(ws);

    ws.on('close', () => {
        clients[datasetId] = clients[datasetId].filter(client => client !== ws);
    });
});

function broadcastProgress(datasetId, message) {
    if (clients[datasetId]) {
        clients[datasetId].forEach(ws => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ message }));
            }
        });
    }
}

app.post('/upload', upload.single('dataset'), async (req, res) => {
    const file = req.file;
    const filename = file.filename;
    const filepath = path.join(file.destination, file.filename);

    const sql = 'INSERT INTO cvr_predictions (filename, status) VALUES (?, "In Progress")';
    const [result] = await db.execute(sql, [filename]);

    const id = result.insertId;

    // Run Python script
    runPythonScript(filepath, id);

    res.redirect('/');
});

app.post('/retry/:id', async (req, res) => {
    const id = req.params.id;
    const [dataset] = await db.query('SELECT * FROM cvr_predictions WHERE id = ?', [id]);

    if (dataset.length > 0) {
        const filepath = path.join('uploads/datasets', dataset[0].filename);
        const updateSql = 'UPDATE cvr_predictions SET status = "In Progress" WHERE id = ?';
        await db.execute(updateSql, [id]);

        // Re-run Python script
        runPythonScript(filepath, id);
    }

    res.redirect('/');
});

function runPythonScript(filepath, id) {
    const spawn = require('child_process').spawn;
    const process = spawn('python3', ['./prediction.py', filepath, id]);

    process.stdout.on('data', async (data) => {
        const message = data.toString().trim();
        console.log(message);

        const updateSql = 'INSERT INTO progress_logs (prediction_id, message) VALUES (?, ?)';
        await db.execute(updateSql, [id, message]);

        broadcastProgress(id, message);
    });

    process.stderr.on('data', async (data) => {
        const error_message = data.toString().trim();
        console.log(error_message);

        const updateSql = 'INSERT INTO progress_logs (prediction_id, message) VALUES (?, ?)';
        await db.execute(updateSql, [id, error_message]);

        broadcastProgress(id, error_message);
    });

    process.on('close', async (code) => {
        const resultPath = `/uploads/results/${id}`; // Assuming the result image is saved with this pattern
        const updateSql = 'UPDATE cvr_predictions SET status = "Finished" WHERE id = ?';
        await db.execute(updateSql, [id]);

        broadcastProgress(id, 'Processing finished.');
    });
}
