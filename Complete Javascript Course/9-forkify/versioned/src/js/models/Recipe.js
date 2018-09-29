import axios from 'axios';

export default class Recipe {
    constructor(id) {
        this.id = id;
    }

    async getRecipe() {
        try {
            const res = await axios(`https://www.food2fork.com/api/get?key=${process.env.API_KEY}&rId=${this.id}`);
            this.title = res.data.recipe.title;
            this.author = res.data.recipe.publisher;
            this.img = res.data.recipe.image_url;
            this.url = res.data.recipe.source_url;
            this.ingredients = res.data.recipe.ingredients;
        } catch (e) {
            throw Error(e);
        }
    }

    calcTime() {
        const numberIngredients = this.ingredients.length;
        const periods = Math.ceil(numberIngredients / 3);
        this.time = periods * 15;
    }

    calcServings() {
        this.servings = 4;
    }

    parseIngredients() {
        const units = {
            "tbsp": ["tablespoons", "tablespoon"],
            "tsp": ["teaspoons", "teaspoon"],
            "oz": ["ounces", "ounce"],
            "lb": ["pounds", "pound", "lbs"],
            "cup": ["cups"],
            'kg': ["kilogram", "kilograms"],
            "g": ["grams", "gram"]
        };

        const newIngredients = this.ingredients.map(el => {
                let ingredient = el.toLowerCase();

                Object.entries(units).forEach(
                    ([key, value]) => {
                        value.forEach(el => {
                            ingredient = ingredient.replace(el, key);
                        })
                    }
                );

                ingredient = ingredient.replace(/ *\([^)]*\) */g, ' ').trim();

                const arrIng = ingredient.split(' ');
                const unitIndex = arrIng.findIndex(el => Object.keys(units).includes(el));

                let objIng;

                if (unitIndex > -1) {
                    const arrCount = arrIng.slice(0, unitIndex);

                    let count;

                    if (arrCount.length === 1) {
                        count = eval(arrIng[0].replace('-', '+'));
                    } else {
                        count = eval(arrIng.slice(0, unitIndex).join('+'));
                    }

                    objIng = {
                        count,
                        unit: arrIng[unitIndex],
                        ingredient: arrIng.slice(unitIndex + 1).join(' ')
                    }
                } else if (parseInt(arrIng[0], 10)) {
                    objIng = {
                        count: parseInt(arrIng[0], 10),
                        unit: '',
                        ingredient: arrIng.slice(1).join(' ')
                    }
                } else if (unitIndex === -1) {
                    objIng = {
                        count: 1,
                        unit: '',
                        ingredient
                    }
                }

                return objIng;
            }
        );

        this.ingredients = newIngredients;
    }

    updateServings(type) {
        const newServings = type === 'dec' ? this.servings - 1 : this.servings + 1;

        this.ingredients.forEach(ing => {
            ing.count *= (newServings / this.servings);
        });

        this.servings = newServings;
    }
}