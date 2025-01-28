const ManualEntryForm = ({ products, setProducts, schema }) => {
    return (
        <div className="space-y-4">
            {products.map((product, index) => (
                <div key={index} className="p-4 border rounded-lg">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="font-medium">Product {index + 1}</h3>
                        {index > 0 && (
                            <button
                                onClick={() => {
                                    const newProducts = products.filter((_, i) => i !== index);
                                    setProducts(newProducts);
                                }}
                                className="text-red-500 hover:text-red-700"
                            >
                                <Trash2 className="h-4 w-4" />
                            </button>
                        )}
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        {/* Required Fields First */}
                        {['id_column', 'name_column'].map(key => (
                            schema[key] && (
                                <div key={key} className="col-span-1">
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        {key.replace('_column', '').split('_').join(' ').toUpperCase()}
                                        <span className="text-red-500 ml-1">*</span>
                                    </label>
                                    <input
                                        type="text"
                                        value={product[schema[key]] || ''}
                                        onChange={(e) => {
                                            const newProducts = [...products];
                                            newProducts[index] = {
                                                ...newProducts[index],
                                                [schema[key]]: e.target.value
                                            };
                                            setProducts(newProducts);
                                        }}
                                        className="w-full p-2 border rounded-md"
                                        required
                                    />
                                </div>
                            )
                        ))}
                        
                        {/* Optional Fields */}
                        {['description_column', 'category_column'].map(key => (
                            schema[key] && (
                                <div key={key} className="col-span-1">
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        {key.replace('_column', '').split('_').join(' ').toUpperCase()}
                                    </label>
                                    <input
                                        type="text"
                                        value={product[schema[key]] || ''}
                                        onChange={(e) => {
                                            const newProducts = [...products];
                                            newProducts[index] = {
                                                ...newProducts[index],
                                                [schema[key]]: e.target.value
                                            };
                                            setProducts(newProducts);
                                        }}
                                        className="w-full p-2 border rounded-md"
                                    />
                                </div>
                            )
                        ))}
                    </div>
                </div>
            ))}
            <button
                onClick={() => setProducts([...products, {}])}
                className="w-full p-2 border-2 border-dashed rounded-lg text-gray-600 hover:text-gray-900 hover:border-gray-400"
            >
                <PlusCircle className="w-4 h-4 inline mr-2" />
                Add Another Product
            </button>
        </div>
    );
};
