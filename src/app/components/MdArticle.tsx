import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import jsx from 'react-syntax-highlighter/dist/esm/languages/prism/jsx';
import js from 'react-syntax-highlighter/dist/esm/languages/prism/javascript';
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';
import { dark } from "react-syntax-highlighter/dist/esm/styles/prism";
import style from './markdown-styles.module.css';


SyntaxHighlighter.registerLanguage('jsx', jsx);
SyntaxHighlighter.registerLanguage('javascript', js);
SyntaxHighlighter.registerLanguage('python', python);

export interface MdArticleProps {
    title: string;
    category: string;
}

function MdArticle({ title, category }: MdArticleProps) {
    const mdPath = require(`../articles/${category}/${title}.md`)
    const [terms, setTerms] = React.useState('')

    const imagesPath = `../assets/article_images/${category}/${title}/`

    React.useEffect(() => {
        fetch(mdPath).then((response) => response.text()).then((text) => {
            setTerms(text)
        })
        // setTerms(markdownTable)
    }, [mdPath, category, title])

    return (
      <div className='grid-container'>
        <div>
          <ReactMarkdown 
            className={style.reactMarkDown} 
            children={terms}
            remarkPlugins={[[remarkGfm, {tableCellPadding: true, tablePipeAlign: true}]]}
            components={{
              code({node, inline, className, children, ...props}) {
                const match = /language-(\w+)/.exec(className || '')
                return !inline && match ? (
                  <SyntaxHighlighter
                    {...props}
                    children={String(children).replace(/\n$/, '')}
                    style={dark}
                    wrapLines={true}
                    language={match[1]}
                    PreTag="div"
                  />
                ) : (
                  <code {...props} className={className}>
                    {children}
                  </code>
                )
              }
            }}/>
        </div>
      </div>
    )
  }

export default MdArticle