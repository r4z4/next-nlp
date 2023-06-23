import db from '../db'
// define a schema for the users table

async function main() {
    db.run(`
        INSERT INTO users (name,email,password)
        VALUES ('aaron','aaron@user.com','password');
    `);
}

main()
  .then(async () => {
    console.log('Inserted :)')
  })
  .catch(async (e) => {
    console.error(e)
    process.exit(1)
  })
