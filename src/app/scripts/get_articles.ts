import db from '../db'
import ArticleProps from '../components/Article'
// define a schema for the users table

interface Article {
  id: number
  title: string
  category: string
  published: boolean
  created_at: Date
  updated_at: Date
}

interface ArticleRows {
  [x: string]: any
  rows: Article[]
}

async function main() {
  const articles = db.all(`
    SELECT * FROM articles;
  `, function(err: any, rows: ArticleRows) {
      let news: Article[] = []
      let trivia: Article[] = []
      rows.map((row: Article) => {
        if (row.category == 'news') {
          news.push(row)
        }
        if (row.category == 'trivia') {
          trivia.push(row)
        }
      })
      console.log(`news = ${news} and trivia = ${trivia}`)
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
