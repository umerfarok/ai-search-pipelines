export default function Home() {
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-4">Welcome to the Playground</h1>
      <p className="text-lg">
        This is a playground for you to test your application training sessions and model versions.
      </p>
      <div className="mt-8">
        <h2 className="text-2xl font-semibold mb-2">Features:</h2>
        <ul className="list-disc list-inside">
          <li>Test application training sessions</li>
          <li>Manage and view model versions</li>
          <li>Upload and process data</li>
        </ul>
      </div>
    </div>
  );
}