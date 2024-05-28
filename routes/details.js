const express = require('express');
const router = express.Router();
const db = require('../db');

router.get('/:id', async (req, res) => {
    const id = req.params.id;
    const [dataset] = await db.query('SELECT * FROM cvr_predictions WHERE id = ?', [id]);
    const [progressLogs] = await db.query('SELECT * FROM progress_logs WHERE prediction_id = ? ORDER BY timestamp', [id]);    
    res.render('detail', { dataset: dataset[0], progressLogs });
});

module.exports = router;
