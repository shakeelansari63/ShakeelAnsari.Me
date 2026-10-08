import { useEffect, useState } from 'react';
import type { LanguageStat } from '../../models/types';
import { fetchCommitLanguages } from '../../services/stats';
import LanguageDonutCard from './LanguageDonutCard';

export default function CommitLanguageCard() {
  const [data, setData] = useState<LanguageStat[] | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCommitLanguages()
      .then(setData)
      .finally(() => setLoading(false));
  }, []);

  return <LanguageDonutCard title="Top Languages by Commit" data={data} loading={loading} />;
}
