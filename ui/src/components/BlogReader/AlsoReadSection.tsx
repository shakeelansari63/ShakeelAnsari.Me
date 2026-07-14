import { useNavigate } from "react-router-dom";
import type { BlogPost } from "../../models/types";

interface Props {
  posts: BlogPost[];
  isLight?: boolean;
}

export default function AlsoReadSection({ posts, isLight }: Props) {
  const navigate = useNavigate();

  if (posts.length === 0) return null;

  return (
    <div className="mt-6">
      <h2
        className={`text-xl font-bold mb-3 ${
          isLight ? "text-pink-600" : "text-pink-400"
        }`}
      >
        Also Read
      </h2>
      {posts.map((post, i) => (
        <div
          key={post.id}
          className={`flex flex-column gap-1 p-3 cursor-pointer ${
            i < posts.length - 1
              ? isLight
                ? "border-bottom-1 border-gray-200"
                : "border-bottom-1 border-gray-700"
              : ""
          }`}
          onClick={() => {
            window.scrollTo(0, 0);
            navigate(`/blog/${post.id}`);
          }}
        >
          <span
            className={`font-bold ${
              isLight ? "text-pink-600" : "text-pink-400"
            }`}
          >
            {post.title}
          </span>
          <p
            className={`text-sm m-0 ${
              isLight ? "text-gray-600" : "text-blue-400"
            }`}
            style={{ fontFamily: '"Space Grotesk", sans-serif' }}
          >
            {post.excerpt}
          </p>
          <span
            className={`text-xs ${
              isLight ? "text-gray-500" : "text-gray-400"
            }`}
          >
            {post.date} &middot; {post.readTime}
          </span>
        </div>
      ))}
    </div>
  );
}
