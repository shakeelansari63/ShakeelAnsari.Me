import { Card } from 'primereact/card';
import { Button } from 'primereact/button';

interface ExpoItem {
  name: string;
  description: string;
  appUrl?: string;
  codeUrl?: string;
  thumbnail: string;
}

interface Props {
  item: ExpoItem;
}

export default function ExpoCard({ item }: Props) {
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
          <div className="flex justify-content-center gap-2">
            {item.appUrl && (
              <Button
                label="App"
                icon="pi pi-external-link"
                rounded
                className="border-gradient-purple text-white"
                onClick={() => window.open(item.appUrl, '_blank')}
                style={{ outline: 'none', boxShadow: 'none' }}
              />
            )}
            {item.codeUrl && (
              <Button
                label="Code"
                icon="pi pi-github"
                rounded
                severity="secondary"
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
