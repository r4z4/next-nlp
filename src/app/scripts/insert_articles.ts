import db from '../db'
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();
//createMany cannot be used w/ SQLite
async function main() {


    db.run(`
        INSERT INTO articles (title,published,filename,url,images,category,createdAt,updatedAt)
        VALUES ('TREC_EDA',FALSE,'TREC_EDA','/articles/trec/trec_eda','run_01.png,run_01_clean.png','trec',datetime('now'),datetime('now'));
    `);

  // let article = await prisma.article.create({
  //   data: {
  //     title: "TREC EDA",
  //     published: true,
  //     createdAt: new Date(),
  //     updatedAt: new Date(),
  //     filename: "TREC_EDA",
  //     url: "/articles/trec/trec_eda",
  //     images: "run_01.png,run_01_clean.png",
  //     category: "trec",
  //   },
  // });

}

main()
  .then(async () => {
    console.log('Inserted :)')
  })
  .catch(async (e) => {
    console.error(e)
    process.exit(1)
  })
