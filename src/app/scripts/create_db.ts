import db from '../db'
// define a schema for the users table


async function main() {
  db.run(`
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        f_name TEXT NOT NULL,
        l_name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_user_id ON users (id, name, email, password);
  `)

  db.run(`
    CREATE TABLE IF NOT EXISTS articles (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      published INTEGER NOT NULL DEFAULT 0,
      images TEXT NOT NULL,
      category TEXT NOT NULL,
      filename TEXT NOT NULL UNIQUE,
      createdAt TEXT,
      updatedAT TEXT,
      url TEXT NOT NULL
  );
  CREATE UNIQUE INDEX IF NOT EXISTS idx_article_id ON articles (id, title, category, filename);
`)
}

main()
  .then(async () => {
    console.log('DB Created :)')
  })
  .catch(async (e) => {
    console.error(e)
    process.exit(1)
  })
