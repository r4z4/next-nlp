import Article, { ArticleProps } from "@/app/components/Article";
import articleStyles from './Article.module.css'
import db from '../../db'

export default async function NewsArticle({
    params,
}: {
    params: {id: string };
}) {
    const article = db.all('SELECT * FROM article WHERE id = ?', params.id, function(err: any, row: ArticleProps) {
        if (err) {
            throw err;
          }
        return row;
    });

    return (
        <div className={articleStyles.card}>
            <div className={articleStyles.cardBody}>
                <h2>Articles</h2>
                <div>
                    <Article article={article} />
                </div>
            </div>
        </div>
    )
}