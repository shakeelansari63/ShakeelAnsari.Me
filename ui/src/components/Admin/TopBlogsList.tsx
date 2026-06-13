import { Card } from "primereact/card";

interface BlogEntry {
    id: string;
    title: string;
    views: number;
    likes: number;
}

interface Props {
    blogs: BlogEntry[];
}

export default function TopBlogsList({ blogs }: Props) {
    return (
        <Card>
            <h3 className="text-white text-lg font-bold mb-3">Top Blogs by Views</h3>
            <div className="flex flex-column gap-2">
                {blogs.map((blog, i) => (
                    <div
                        key={blog.id}
                        className="flex align-items-center justify-content-between"
                        style={{ padding: "0.5rem 0", borderBottom: "1px solid #333" }}
                    >
                        <div className="flex align-items-center gap-3">
                            <span className="text-gray-500 text-sm">{i + 1}.</span>
                            <span className="text-pink-400">{blog.title || blog.id}</span>
                        </div>
                        <div className="flex gap-3 text-sm">
                            <span className="text-gray-400"><i className="pi pi-eye mr-1" />{blog.views}</span>
                            <span className="text-gray-400"><i className="pi pi-heart mr-1" />{blog.likes}</span>
                        </div>
                    </div>
                ))}
            </div>
        </Card>
    );
}
