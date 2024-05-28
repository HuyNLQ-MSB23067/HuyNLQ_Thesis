const express = require('express');
const router = express.Router();
const db = require('../db');

router.get('/', async (req, res) => {
    const [datasets] = await db.query('SELECT * FROM cvr_predictions ORDER BY id DESC');
    res.render('index', { datasets });
});

module.exports = router;
