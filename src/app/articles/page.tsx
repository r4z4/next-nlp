import Article, { ArticleProps } from "@/app/components/Article";
import homeStyles from '../components/Home.module.css'
import db from '@/app/db';
import CollapsePanel from "../components/CollapsePanel";
import { getArticles } from './actions'

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

export default async function ArticleHome({
    params,
}: {
    params: {id: number };
}) {

    const grouped = [{articles: [{id: 1, category: 'news', title: 'gg', published: false, createdAt: '', updatedAt: ''}]},{articles: []}]

    return (
        <div className={homeStyles.card}>
            <div className={homeStyles.cardBody}>
                <h2>Articles</h2>
                <div>
                    <ul>
                        {grouped.map((category: CategoryProps) => (
                            <li>
                                <CollapsePanel panelData={getPanelData(category)} /> 
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        </div>
    )
}