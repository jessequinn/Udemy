import axios from 'axios';

export default class Search {
    constructor(query) {
        this.query = query;
    }

    /**
     * All search requests should be made to the search API URL.
     *
     * https://www.food2fork.com/api/search
     *
     * All recipe requests should be made to the recipe details API URL.
     *
     * https://www.food2fork.com/api/get
     */
    async getResults() {
        // const proxy = 'https://cors-anywhere.herokuapp.com/'
        // const res = await axios(`${proxy}https://www.food2fork.com/api/search?key=${process.env.API_KEY}&q=${query}`);
        try {
            const res = await axios(`https://www.food2fork.com/api/search?key=${process.env.API_KEY}&q=${this.query}`);
            this.result = res.data.recipes;
        } catch (e) {
            throw Error(e);
        }
    }
}