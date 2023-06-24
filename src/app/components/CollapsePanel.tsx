import React from 'react'
import CollapseClient from './CollapseClient'

export interface CollapsePanelProps {
    panelData: PanelData;
}

export interface PanelData {
    name: String;
    date: String;
    desc: String;
    img?: string;
    bgColor?: string;
    category: String;
    documents: PanelDocument[];
}

export interface PanelDocument {
    id: number;
    filename: String;
    url: string;
    previewComponent: JSX.Element;
}

// const panelData {
//     name = 'Panel Name',
//     date = '05-22-2020',
//     category = 'Government',
//     documents = ['Doc1', 'Doc2', 'Doc3']
// }

function CollapsePanel({ panelData }: CollapsePanelProps) {

  return (
    <div className="collapse-panel" style={{backgroundColor: panelData.bgColor ? panelData.bgColor : ''}}>
      <CollapseClient panelData={panelData}/>
    </div>
  );
}

export default CollapsePanel;