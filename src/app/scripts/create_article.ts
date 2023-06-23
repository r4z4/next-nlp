import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();
//createMany cannot be used w/ SQLite
async function main() {
  let article = await prisma.article.create({
    data: {
      title: "TREC EDA",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "TREC_EDA",
      url: "/articles/trec/trec_eda",
      images: "run_01.png,run_01_clean.png",
      category: "trec",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "TREC Aug",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "TREC_AUG",
      url: "/articles/trec/trec_aug",
      images: "run_01.png,run_01_clean.png",
      category: "trec",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Run 01",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "Run 01",
      url: "/articles/trec/run_01",
      images: "run_01.png,run_01_clean.png",
      category: "trec",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Run 02",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "Run 02",
      url: "/articles/trec/run_02",
      images: "run_01.png,run_01_clean.png",
      category: "trec",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Run 01",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "Run 01",
      url: "/articles/glove/run_01",
      images: "run_01.png,run_01_clean.png",
      category: "glove",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Run 02",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "Run 02",
      url: "/articles/glove/run_02",
      images: "run_01.png,run_01_clean.png",
      category: "glove",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Run 03",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "Run 01",
      url: "/articles/glove/run_03",
      images: "run_03.png,run_03_clean.png",
      category: "glove",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Run 04",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "Run 02",
      url: "/articles/glove/run_04",
      images: "run_04.png,run_04_clean.png",
      category: "glove",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Transformers",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "01_Transformers",
      url: "/articles/topic-modeling/01_transformers",
      images: "run_01.png,run_01_clean.png",
      category: "topic_modeling",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "LDA",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "01_LDA",
      url: "/articles/topic-modeling/01_LDA",
      images: "run_01.png,run_01_clean.png",
      category: "topic_modeling",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Trivia LDA",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "LDA_Trivia",
      url: "/articles/trivia/lda_trivia",
      images: "run_01.png,run_01_clean.png",
      category: "trivia",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Generate",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "Generate",
      url: "/articles/embeddings/generate",
      images: "run_01.png,run_01_clean.png",
      category: "generate",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Embedding Visualizations",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "Embedding Visualizations",
      url: "/articles/dimred/viz",
      images: "run_01.png,run_01_clean.png",
      category: "dimred",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Newsgroup EDA",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "NewsgroupEDA",
      url: "/articles/news/eda",
      images: "run_01.png,run_01_clean.png",
      category: "news",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Run 01",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "20_News Run 01",
      url: "/articles/news/clean_run_01",
      images: "run_01.png,run_01_clean.png",
      category: "news",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Run 02",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "20_News Run 02",
      url: "/articles/news/clean_run_02",
      images: "run_02.png,run_02_clean.png",
      category: "news",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Run 03",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "20_News Run 03",
      url: "/articles/news/clean_run_03",
      images: "run_03.png,run_03_clean.png",
      category: "news",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Run 04",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "20_News Run 04",
      url: "/articles/news/clean_run_04",
      images: "run_04.png,run_04_clean.png",
      category: "news",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Run 01",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "20_News Run 01",
      url: "/articles/news/body_clean_run_01",
      images: "run_01.png,run_01_clean.png",
      category: "news",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Run 02",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "20_News Run 02",
      url: "/articles/news/body_clean_run_02",
      images: "run_02.png,run_02_clean.png",
      category: "news",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Run 03",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "20_News Run 03",
      url: "/articles/news/body_clean_run_03",
      images: "run_03.png,run_03_clean.png",
      category: "news",
    },
  });

  article = await prisma.article.create({
    data: {
      title: "Run 04",
      published: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      filename: "20_News Run 04",
      url: "/articles/news/body_clean_run_04",
      images: "run_04.png,run_04_clean.png",
      category: "news",
    },
  });

  console.log(article);
}

main()
  .then(async () => {
    await prisma.$disconnect();
  })
  .catch(async (e) => {
    console.error(e);
    await prisma.$disconnect();
    process.exit(1);
  });
