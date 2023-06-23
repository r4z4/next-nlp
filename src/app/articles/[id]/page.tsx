import MdArticle, { MdArticleProps } from "@/app/components/MdArticle";
import homeStyles from '../components/Home.module.css'
import db from '@/app/db';

interface ArticleQueryRow {
    id: number
    title: string
    category: string
}

export default async function Article({
    params,
}: {
    params: {id: number };
}) {
    const article = await db.all(`SELECT title, category FROM articles WHERE id = ?`, params.id, function(err: any, row: ArticleQueryRow) {
        if (err) {
            throw err;
          }
        return row;
    }); 
    const grouped = await db.run(`INSERT INTO`);

    return (
        <div className={homeStyles.card}>
            <div className={homeStyles.cardBody}>
                <div>
                    <MdArticle title={article.title} category={article.category} /> 
                </div>
            </div>
        </div>
    )
}