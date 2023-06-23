import db from '../db'
// define a schema for the users table


async function main() {
  db.run(`
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    );
    CREATE UNIQUE INDEX idx_id ON users (id, name, email, password);
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
