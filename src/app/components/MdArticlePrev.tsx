import React from 'react'

interface MdArticlePrevProps {
    title: string
    desc: string
    concepts: string[]
    hangups: string[]
}

function MdArticle({ title, desc, concepts, hangups }: MdArticlePrevProps) {

    // const [terms, setTerms] = React.useState('')

    // React.useEffect(() => {
    //     fetch(mdPath).then((response) => response.text()).then((text) => {
            
    //         setTerms(text)
    //     })
    //     // setTerms(markdownTable)
    // }, [])

  return (
    <div className={'prev-container'}>
      <h4 className={'prev-title'}>{title}</h4>
      <div className={'prev-body'}>
        <span className={'byline'}>
            <p>{desc}</p>
        </span>
      <div className={'prev-list'}>
        <h6>Concepts</h6>
        <ul>
          {concepts.map((concept: string, index: number)=><li key={index}>{concept}</li>)}
        </ul>
      </div>
      <div className={'prev-list'}>
        <h6>HangUps</h6>
        <ul>
          {hangups.map((hangup: string, index: number)=><li key={index}>{hangup}</li>)}
        </ul>
      </div>
      </div>
    </div>
    )
  }

export default MdArticle