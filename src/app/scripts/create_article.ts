import { PrismaClient } from '@prisma/client'
import { randomUUID } from 'crypto'

const prisma = new PrismaClient()

async function main() {
    const article = await prisma.article.create({
        data: {
            id: randomUUID(),
            title: 'Test Article One',
            subDir: 'alice@prisma.io',
            complete: false,
            createdAt: new Date(),
            updatedAt: new Date(),
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