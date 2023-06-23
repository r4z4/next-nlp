import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()
//createMany cannot be used w/ SQLite
async function main() {
    let article = await prisma.article.create({
        data: {
            title: 'Test Article One',
            published: true,
            createdAt: new Date(),
            updatedAt: new Date(),
            url: '/articles/run_01',
            filename: 'run_01',
            images: 'run_01.png,run_01_clean.png',
            category: 'news',
        },
        })
    
    article = await prisma.article.create({
      data: {
          title: 'Test Article Two',
          published: true,
          createdAt: new Date(),
          updatedAt: new Date(),
          url: '/articles/run_01',
          filename: 'run_01',
          images: 'run_01.png,run_01_clean.png',
          category: 'trivia',
      },
      })

    article = await prisma.article.create({
      data: {
          title: 'Test Article Three',
          published: true,
          createdAt: new Date(),
          updatedAt: new Date(),
          url: '/articles/run_01',
          filename: 'run_01',
          images: 'run_01.png,run_01_clean.png',
          category: 'trivia',
      },
      })

    article = await prisma.article.create({
      data: {
          title: 'Test Article Four',
          published: true,
          createdAt: new Date(),
          updatedAt: new Date(),
          url: '/articles/run_01',
          filename: 'run_01',
          images: 'run_01.png,run_01_clean.png',
          category: 'news',
      },
      })

        console.log(article)
}

main()
  .then(async () => {
    await prisma.$disconnect()
  })
  .catch(async (e) => {
    console.error(e)
    await prisma.$disconnect()
    process.exit(1)
  })

