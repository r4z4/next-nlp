"use client"
import {Glove_01, Glove_02, Glove_03, Glove_04} from '../../assets/mdx/glove/'
import remarkGfm from 'remark-gfm'
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import jsx from 'react-syntax-highlighter/dist/esm/languages/prism/jsx';
import js from 'react-syntax-highlighter/dist/esm/languages/prism/javascript';
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';
import { dark } from "react-syntax-highlighter/dist/esm/styles/prism";
import mdxStyles from '../../components/Mdx.module.css'

SyntaxHighlighter.registerLanguage('jsx', jsx);
SyntaxHighlighter.registerLanguage('javascript', js);
SyntaxHighlighter.registerLanguage('python', python);

type MapType = { 
    [id: string]: string; 
}

interface CodeFnProps {
    className: string
    children: string
}

export default async function MdxArticle(id: number) {
    function code({className, children, ...props}: CodeFnProps) {
        const match = /language-(\w+)/.exec(className || '')
        return match
          ? <SyntaxHighlighter 
                remarkPlugins={[[remarkGfm, {tableCellPadding: true, tablePipeAlign: true}]]} 
                children={String(children).replace(/\n$/, '')} 
                language={match[1]} 
                style={dark} 
                wrapLines={true}
                PreTag="div" 
                {...props} 
            />
          : <code className={className} {...props} />
      }
    function getMdx(id: number) {
        
        if (id == 1) {
            return <Glove_01 className={mdxStyles.reactMarkDown} components={{code}} />
        } else if (id == 2) {
            return <Glove_02 className={mdxStyles.reactMarkDown} components={{code}} />
        } else if (id == 3) {
            return <Glove_03 className={mdxStyles.reactMarkDown} components={{code}} />
        } else if (id == 4) {
            return <Glove_04 className={mdxStyles.reactMarkDown} components={{code}} />
        } else {
            return <Glove_04 className={mdxStyles.reactMarkDown} components={{code}} />
        }
    }

    return (
        <div className={mdxStyles.reactMarkDown}>
            {getMdx(id)}
        </div>
    )
}