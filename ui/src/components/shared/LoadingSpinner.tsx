import { Spin, Typography } from 'antd';

export default function LoadingSpinner() {
  const { Text } = Typography;

  return (
    <div className="flex flex-col justify-center items-center" style={{ minHeight: '200px', gap: 12 }}>
      <Spin size="large" />
      <Text type="secondary">Loading...</Text>
    </div>
  );
}