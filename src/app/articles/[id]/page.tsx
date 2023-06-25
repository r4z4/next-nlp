"use client"
import Mdx from '../../assets/md/glove/run_04.mdx'
import {Glove_01, Glove_02, Glove_03, Glove_04} from '../../assets/mdx/glove/'
import ArticleImage from '../assets/article_images/glove/run_04.png'

type MapType = { 
    [id: string]: string; 
}

export default async function MdxArticle({
    params,
}: {
    params: {id: number };
}) {
    console.log(`id=${params.id}`)
    function getMdx(id: number) {
        if (id == 1) {
            return <Glove_03 />
        } else if (id == 2) {
            return <Glove_04 />
        } else {
            return <Glove_04 />
        }
    }
    const mdx = () => getMdx(params.id)
    return (
        <div>
            <p>Hey</p>
            {getMdx(params.id)}
        </div>
    )
}