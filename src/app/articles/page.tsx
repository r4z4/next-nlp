import Article, { ArticleProps } from "@/app/components/Article";
import homeStyles from '../components/Home.module.css'
import db from '@/app/db';

export default async function ArticleHome({
    params,
}: {
    params: {id: number };
}) {
    const articles = await db.run(`INSERT INTO`);
    const grouped = await db.run(`INSERT INTO`);

    return (
        <div className={homeStyles.card}>
            <div className={homeStyles.cardBody}>
                <h2>Articles</h2>
                <div>
                    <ul>
                        {(await articles).map((article: ArticleProps) => (
                            <li>
                                <Article article={article} /> 
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
            <div className={homeStyles.cardBody}>
                <h2>Articles Grouped</h2>
                <div>
                    <ul>
                        {(await grouped).map((group: any) => (
                            <ul>
                                {group.map((item: any) => (
                                    <li>
                                        {JSON.stringify(item)}
                                    </li>
                                ))} 
                            </ul>
                        ))}
                    </ul>
                </div>
            </div>
        </div>
    )
}