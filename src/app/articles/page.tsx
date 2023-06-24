import Article, { ArticleProps } from "@/app/components/Article";
import homeStyles from '../components/Home.module.css'
import db from '@/app/db';
import CollapseData from "../components/CollapseData";
import { getArticles } from './actions'

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

interface CategoryProps {
    articles: ArticleProps[]
}

function getPanelData(category: CategoryProps) {
    return {
        name: "Test Run",
        date: "05/30/2020",
        desc: "Test Desc",
        img: '',
        bgColor: "#FFFFFF",
        category: "trivia",
        documents: []
    }
}

function getData(callback: { (q: any): void; (arg0: string | any[] | ArticleRows): void; }) {
    const res = db.all(`
        SELECT * FROM articles;
        `, function(err: any, rows: ArticleRows) {
        let news: Article[] = []
        let trivia: Article[] = []

        console.log(`Rows => ${JSON.stringify((rows))}`)

        rows.map((row: Article) => {
            if (row.category == 'news') {
            news.push(row)
            }
            if (row.category == 'trivia') {
            trivia.push(row)
            }
        })
        console.log(`Rows Pre Return: ${JSON.stringify(rows)}`)
        callback(returnData(rows))
    })
}

function returnData(data: string | any[] | ArticleRows){
    console.log(data.length); // 3
    return data;
}

export default async function ArticleHome({
    params,
}: {
    params: {id: number };
}) {

    let data: any;
    let newsData: any;
    let triviaData: any;
    let gloveData: any;
    let trecData: any;
    let tmData: any;

    getData(function(q) {
        let news: Article[] = []
        let trivia: Article[] = []
        let trec: Article[] = []
        let glove: Article[] = []
        let tm: Article[] = []
        console.log(`Rows => ${JSON.stringify((q))}`)

        q.map((row: Article) => {
            switch (row.category) {
                case 'news':
                    news.push(row)
                    break;
                case 'trivia':
                    trivia.push(row)
                    break;
                case 'glove':
                    glove.push(row)
                    break;
                case 'topic-modeling':
                    tm.push(row)
                    break;
                case 'trec':
                    trec.push(row)
                    break;
            }
        })

        newsData = news
        triviaData = trivia
        trecData = trec
        tmData = tm
        gloveData = glove
    });

    console.log(`Data ==== ${JSON.stringify(newsData)}`)

    const test = async () => {
        await new Promise(r => setTimeout(r, 5000));
        console.log(`Delayed Data: ${JSON.stringify(data)}`)
    }

    const huh = async () => {
        await new Promise(r => setTimeout(r, 5000));
        return 'blouse'}

    test()

    const grouped = [{articles: [{id: 1, category: 'news', title: 'gg', published: false, createdAt: '', updatedAt: ''}]},{articles: []}]

    return (
        <div className={homeStyles.card}>
            <div className={homeStyles.cardBody}>
                <h2>Articles</h2>
                <p>{await huh() ? 'e' : 'f'}</p>
                <div>
                    <p>{JSON.stringify(newsData)}</p>
                    <CollapseData data={[
                        {name: 'news', articles: newsData}, 
                        {name: 'trivia', articles: triviaData},
                        {name: 'glove', articles: gloveData},
                        {name: 'trec', articles: trecData},
                        {name: 'tm', articles: tmData}
                    ]}
                    />
                </div>
            </div>
        </div>
    )
}