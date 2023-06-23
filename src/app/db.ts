const sqlite3 = require('sqlite3').verbose();

// create a new database
const db = new sqlite3.Database('dev.sqlite');

export default db;