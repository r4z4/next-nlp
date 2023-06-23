import { article01, article02, article03 } from '../utils/articleList'

var sqlite3 = require('sqlite3').verbose(),
    db = new sqlite3.Database('dev.sqlite');

interface UserQueryRequest {
    first?: string
    last?: string
    email?: string
}

interface UserQueryRow {
    id?: string
    f_name?: string
    l_name?: string
}

var runQuery = function(request: UserQueryRequest) {
    console.log(`request = ${JSON.stringify(request)}`)
    var select = `SELECT id, f_name, l_name FROM users WHERE `,
        query,
        params = []; 

    if (request.first && request.last) {
        query = select + 'f_name = ? AND l_name = ?'
        params.push( request.first, request.last );
    } else if (request.first) {
        query = select + 'f_name = ?'
        params.push( request.first );
    } else if (request.last) {
        query = select + `l_name = ?`
        params.push( request.last );
    }   

    if (request.email) {
        params.push( request.email );

        if (!query) {
            query = select + `email like ?`; 
        } else {
            query = query + ` and email like ?`; 
        }   
    }   

    console.log(`query = ${query}`)
    console.log(`params = ${params}`)

    db.all( query, params, function(err: any, row: UserQueryRow) {
        if (err) {
            throw err;
          }
        console.log(JSON.stringify(row));
    }); 
};

const createDBs = () => {
    db.run(`
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        f_name TEXT NOT NULL,
        l_name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_user_id ON users (id, f_name, l_name, email, password);
  `)

    db.run(`
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        published INTEGER NOT NULL DEFAULT 0,
        images TEXT NOT NULL,
        category TEXT NOT NULL,
        filename TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL default CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL default CURRENT_TIMESTAMP,
        url TEXT NOT NULL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_article_id ON articles (id, title, category, filename);
    `)
}

const createTestData = function() {

    var stmt = db.prepare(`INSERT INTO users (f_name, l_name, email, password) VALUES (?, ?, ?, ?);`);

    stmt.run('john', 'smith', 'jon@smith.com', 'password');
    stmt.run('jane', 'doe', 'jane@die.com', 'password');
    stmt.run('jeff', 'lebowski', 'jeff@lebowski.com', 'password');

    stmt.finalize();
};

const createArticleData = function() {

    var stmt = db.prepare(`INSERT INTO articles (title, category, published, images, filename, url) VALUES (?, ?, ?, ?, ?, ?);`);

    console.log(`article01 = ${article01}`)
    // stmt.run('Title01', 'news', 0, '/run01.png', 'filename', 'test_url');
    // Destructure the list so we get them as separate args
    // Turns out we can actually just use the list without destructuring. Leaving in though => good reminder.
    stmt.run(...article01);
    stmt.run(...article02);
    stmt.run(...article03);

    stmt.finalize();
};

db.serialize(function() {
    createDBs();
    createTestData();
    createArticleData();

    var searches = [ 
        { first:'john', last:'smith' },
        { first:'jane' },
        { last:'lebowski' },
        { email:'%.com%' }
    ];  

    searches.forEach(function(request) {
        runQuery( request );
    }); 
});

db.close();