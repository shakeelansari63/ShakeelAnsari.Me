import { useNavigate } from 'react-router-dom';
import { Card } from 'primereact/card';
import { Button } from 'primereact/button';
import type { ExpoProject } from '../../data/expo';

interface Props {
  item: ExpoProject;
}

export default function ExpoCard({ item }: Props) {
  const navigate = useNavigate();

  return (
    <div className="md:col-6 col-12">
      <Card className="h-full" pt={{ body: { className: 'flex flex-column h-full' }, content: { className: 'flex flex-column flex-1' } }}>
        {item.thumbnail && (
          <div className="flex justify-content-center mb-3">
            <img src={item.thumbnail} alt={item.name} style={{ maxWidth: '100%', borderRadius: '8px', maxHeight: '200px', objectFit: 'cover' }} />
          </div>
        )}
        <div className="flex flex-column flex-1">
          <span className="font-bold text-xl text-pink-400 mb-2">{item.name}</span>
          <p className="text-blue-400 m-0 mb-3 flex-1" style={{ fontFamily: '"Space Grotesk", sans-serif' }}>{item.description}</p>
          <div className="expo-card-buttons flex justify-content-center gap-3">
            {item.appUrl && (
              <Button
                label="App"
                icon="pi pi-external-link"
                rounded
                className="border-gradient-purple text-white expo-btn"
                onClick={() => window.open(item.appUrl, '_blank')}
                style={{ outline: 'none', boxShadow: 'none' }}
              />
            )}
            {item.productPageUrl && (
              <Button
                label="Details"
                icon="pi pi-info-circle"
                rounded
                severity="info"
                className="expo-btn"
                onClick={() => navigate(item.productPageUrl!)}
                style={{ outline: 'none', boxShadow: 'none' }}
              />
            )}
            {item.codeUrl && (
              <Button
                label="Code"
                icon="pi pi-github"
                rounded
                severity="secondary"
                className="expo-btn"
                onClick={() => window.open(item.codeUrl, '_blank')}
                style={{ outline: 'none', boxShadow: 'none' }}
              />
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}
