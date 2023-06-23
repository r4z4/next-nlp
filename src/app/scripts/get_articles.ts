import db from '../db'
// define a schema for the users table


async function main() {
  let users = 'what'
  users = db.all(`
    SELECT name FROM users WHERE id = 1;
  `, function(err: any, rows: any) {
      console.log(rows)
  })
}

main()
  .then(() => {
    console.log(`Retrieved :)`)
  })
  .catch(async (e) => {
    console.error(e)
    process.exit(1)
  })
