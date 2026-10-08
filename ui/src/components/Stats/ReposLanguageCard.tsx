import { useEffect, useState } from 'react';
import type { LanguageStat } from '../../models/types';
import { fetchRepoLanguages } from '../../services/stats';
import LanguageDonutCard from './LanguageDonutCard';

export default function ReposLanguageCard() {
  const [data, setData] = useState<LanguageStat[] | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchRepoLanguages()
      .then(setData)
      .finally(() => setLoading(false));
  }, []);

  return <LanguageDonutCard title="Top Languages by Repo" data={data} loading={loading} />;
}
