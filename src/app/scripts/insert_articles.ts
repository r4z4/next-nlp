import db from '../db'

async function main() {


    db.run(`
        INSERT INTO articles (title,published,filename,url,images,category,created_at,updated_at)
        -- TREC
        VALUES ('TREC_EDA',FALSE,'TREC_EDA','/articles/trec/trec_eda','run_01.png,run_01_clean.png','trec',datetime('now'),datetime('now')),
        ('TREC_AUG',FALSE,'TREC_AUG','/articles/trec/trec_aug','run_01.png,run_01_clean.png','trec',datetime('now'),datetime('now')),
        ('TREC_Run_01',FALSE,'TREC_Run_01','/articles/trec/run_01','run_01.png,run_01_clean.png','trec',datetime('now'),datetime('now')),
        ('TREC_Run_02',FALSE,'TREC_Run_02','/articles/trec/run_02','run_01.png,run_01_clean.png','trec',datetime('now'),datetime('now')),
        -- GloVe
        ('GloVe_Run_01',FALSE,'GloVe_Run_01','/articles/glove/run_01','run_01.png','glove',datetime('now'),datetime('now')),
        ('GloVe_Run_02',FALSE,'GloVe_Run_02','/articles/glove/run_02','run_02.png','glove',datetime('now'),datetime('now')),
        ('GloVe_Run_03',FALSE,'GloVe_Run_03','/articles/glove/run_03','run_03.png','glove',datetime('now'),datetime('now')),
        ('GloVe_Run_04',FALSE,'GloVe_Run_04','/articles/glove/run_04','run_04.png','glove',datetime('now'),datetime('now')),
        -- Topic Modeling
        ('01_Transformers',FALSE,'01_Transformers','/articles/topic-modeling/01_Transformers','01_Transformers.png','topic-modeling',datetime('now'),datetime('now')),
        ('02_LDA',FALSE,'02_LDA','/articles/topic-modeling/02_LDA','02_LDA.png','topic-modeling',datetime('now'),datetime('now')),
        -- Trivia
        ('LDA_Trivia',FALSE,'LDA_Trivia','/articles/trivia/LDA_Trivia','LDA_Trivia.png','trivia',datetime('now'),datetime('now'));
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
